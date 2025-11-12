import re
import aiohttp

from astrbot.api.all import *

TEMP_PATH = os.path.abspath("data/temp")


class SDGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.session = None
        self._validate_config()
        os.makedirs(TEMP_PATH, exist_ok=True)

        # 初始化并发控制
        self.active_tasks = 0
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 10)  # 设定最大并发数
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # --- 初始化排队号计数器和锁 ---
        self.queue_counter = 0
        self.queue_lock = asyncio.Lock()

        # 优化：添加资源缓存
        self.resource_cache = {}

    def _validate_config(self):
        """配置验证"""
        self.config["webui_url"] = self.config["webui_url"].strip()
        if not self.config["webui_url"].startswith(("http://", "https://")):
            raise ValueError("WebUI地址必须以http://或https://开头")

        if self.config["webui_url"].endswith("/"):
            self.config["webui_url"] = self.config["webui_url"].rstrip("/")
            self.config.save_config()

    async def ensure_session(self):
        """确保会话连接"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.config.get("session_timeout_time", 120))
            )

    async def _fetch_webui_resource(self, resource_type: str) -> list:
        """从 WebUI API 获取指定类型的资源列表 (带缓存)"""

        # 优化：检查缓存
        if resource_type in self.resource_cache:
            logger.debug(f"从缓存加载 {resource_type} 资源")
            return self.resource_cache[resource_type]

        endpoint_map = {
            "model": "/sdapi/v1/sd-models",
            "embedding": "/sdapi/v1/embeddings",
            "lora": "/sdapi/v1/loras",
            "sampler": "/sdapi/v1/samplers",
            "upscaler": "/sdapi/v1/upscalers",
            "vae": "/sdapi/v1/sd-vae"  # 新增：VAE 接口
        }
        if resource_type not in endpoint_map:
            logger.error(f"无效的资源类型: {resource_type}")
            return []

        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}{endpoint_map[resource_type]}") as resp:
                if resp.status == 200:
                    resources = await resp.json()
                    resource_names = []

                    # 按不同类型解析返回数据
                    if resource_type in ["model", "vae"]:
                        resource_names = [r["model_name"] for r in resources if "model_name" in r]
                    elif resource_type == "embedding":
                        resource_names = list(resources.get('loaded', {}).keys())
                    elif resource_type in ["lora", "sampler", "upscaler"]:
                        resource_names = [r["name"] for r in resources if "name" in r]

                    logger.debug(f"从 WebUI 获取到的{resource_type}资源: {resource_names}")

                    # 优化：存入缓存
                    self.resource_cache[resource_type] = resource_names
                    return resource_names
        except Exception as e:
            logger.error(f"获取 {resource_type} 类型资源失败: {e}")

        return []

    async def _get_sd_model_list(self):
        return await self._fetch_webui_resource("model")

    async def _get_embedding_list(self):
        return await self._fetch_webui_resource("embedding")

    async def _get_lora_list(self):
        return await self._fetch_webui_resource("lora")

    async def _get_sampler_list(self):
        """获取可用的采样器列表"""
        return await self._fetch_webui_resource("sampler")

    async def _get_upscaler_list(self):
        """获取可用的上采样算法列表"""
        return await self._fetch_webui_resource("upscaler")

    async def _get_vae_list(self):
        """获取可用的 VAE 列表"""
        return await self._fetch_webui_resource("vae")

    async def _generate_payload(self, prompt: str) -> dict:
        """
        优化：构建生成参数 (实现原生 Hires. fix)
        """
        params = self.config["default_params"]

        # 基础 payload
        payload = {
            "prompt": prompt,
            "negative_prompt": self.config["negative_prompt_global"],
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "cfg_scale": params["cfg_scale"],
            "batch_size": params["batch_size"],
            "n_iter": params["n_iter"],
            "seed": params.get("seed", -1),
        }

        # 检查是否启用 "高分修复" (Hires. fix)
        if self.config.get("enable_upscale", False):
            # API 文档 (StableDiffusionProcessingTxt2Img)
            # 要求我们添加 Hires. fix 特定参数
            hr_params = {
                "enable_hr": True,
                "hr_scale": params.get("upscale_factor", 2),  # 对应配置中的 "upscale_factor"
                "hr_upscaler": params.get("upscaler", "Latent"),
                "hr_second_pass_steps": params.get("hr_second_pass_steps", 10),
                "denoising_strength": params.get("denoising_strength", 0.4)
            }
            payload.update(hr_params)
            logger.debug(f"Hires. fix 已启用, 添加参数: {hr_params}")

        # 添加 override_settings (用于 Clip Skip 和 VAE)
        override_settings = {
            "CLIP_stop_at_last_layers": params.get("clip_skip", 2),
            "sd_vae": params.get("sd_vae", "Automatic")
        }
        payload["override_settings"] = override_settings
        logger.debug(f"Override settings: {override_settings}")

        return payload

    def _trans_prompt(self, prompt: str) -> str:
        """
        将提示词中的“用于替代空格的字符”替换为为空格
        """
        replace_space = self.config.get("replace_space")
        return prompt.replace(replace_space, " ")

    async def _generate_prompt(self, prompt: str) -> str:
        provider = self.context.get_using_provider()
        if provider:
            prompt_guidelines = self.config["prompt_guidelines"]
            prompt_generate_text = (
                "请根据以下描述生成用于 Stable Diffusion WebUI 的英文提示词，"
                "请返回一条逗号分隔的 `prompt` 英文字符串，适用于 Stable Diffusion web UI，"
                "其中应包含主体、风格、光照、色彩等方面的描述，"
                "避免解释性文本，不需要 “prompt:” 等内容，不需要双引号包裹，"
                "直接返回 `prompt`，不要加任何额外说明。"
                f"{prompt_guidelines}\n"
                "描述："
            )

            response = await provider.text_chat(f"{prompt_generate_text} {prompt}", session_id=None)
            if response.completion_text:
                generated_prompt = re.sub(r"<think>[\s\S]*</think>", "", response.completion_text).strip()
                return generated_prompt

        return ""

    async def _call_sd_api(self, endpoint: str, payload: dict) -> dict:
        """通用API调用函数"""
        await self.ensure_session()
        try:
            async with self.session.post(
                    f"{self.config['webui_url']}{endpoint}",
                    json=payload
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise ConnectionError(f"API错误 ({resp.status}): {error}")
                return await resp.json()
        except aiohttp.ClientError as e:
            raise ConnectionError(f"连接失败: {str(e)}")

    async def _call_t2i_api(self, prompt: str) -> dict:
        """调用 Stable Diffusion 文生图 API"""
        await self.ensure_session()
        payload = await self._generate_payload(prompt)
        return await self._call_sd_api("/sdapi/v1/txt2img", payload)

    # 优化：移除 _apply_image_processing 函数，Hires. fix 已在 _generate_payload 中处理

    async def _set_model(self, model_name: str) -> bool:
        """设置图像生成模型，并存入 config"""
        try:
            # 优化：使用 /sdapi/v1/options 接口设置模型
            async with self.session.post(
                    f"{self.config['webui_url']}/sdapi/v1/options",
                    json={"sd_model_checkpoint": model_name}
            ) as resp:
                if resp.status == 200:
                    self.config["base_model"] = model_name  # 存入 config
                    self.config.save_config()

                    logger.debug(f"模型已设置为: {model_name}")
                    return True
                else:
                    logger.error(f"设置模型失败 (状态码: {resp.status})")
                    return False
        except Exception as e:
            logger.error(f"设置模型异常: {e}")
            return False

    async def _check_webui_available(self) -> (bool, str):
        """服务状态检查"""
        try:
            await self.ensure_session()
            # 优化：使用 /internal/ping 接口检查 (更快)
            async with self.session.get(f"{self.config['webui_url']}/internal/ping") as resp:
                if resp.status == 200:
                    return True, 0
                else:
                    logger.debug(f"⚠️ Stable diffusion Webui 返回值异常，状态码: {resp.status})")
                    return False, resp.status
        except Exception as e:
            logger.debug(f"❌ 测试连接 Stable diffusion Webui 失败，报错：{e}")
            return False, 0

    def _get_generation_params(self) -> str:
        """获取当前图像生成的参数"""
        positive_prompt_global = self.config.get("positive_prompt_global", "")
        negative_prompt_global = self.config.get("negative_prompt_global", "")

        params = self.config.get("default_params", {})
        width = params.get("width") or "未设置"
        height = params.get("height") or "未设置"
        steps = params.get("steps") or "未设置"
        sampler = params.get("sampler") or "未设置"
        cfg_scale = params.get("cfg_scale") or "未设置"
        batch_size = params.get("batch_size") or "未设置"
        n_iter = params.get("n_iter") or "未设置"

        # 新增
        seed = params.get("seed", -1)
        clip_skip = params.get("clip_skip", 2)
        sd_vae = params.get("sd_vae", "Automatic")
        base_model = self.config.get("base_model").strip() or "未设置"

        return (
            f"- 全局正面提示词: {positive_prompt_global}\n"
            f"- 全局负面提示词: {negative_prompt_global}\n"
            f"- 基础模型: {base_model}\n"
            f"- VAE: {sd_vae}\n"
            f"- 图片尺寸: {width}x{height}\n"
            f"- 步数: {steps}\n"
            f"- 采样器: {sampler}\n"
            f"- CFG比例: {cfg_scale}\n"
            f"- 种子: {seed}\n"
            f"- Clip Skip: {clip_skip}\n"
            f"- 批数量: {batch_size}\n"
            f"- 迭代次数: {n_iter}"
        )

    def _get_upscale_params(self) -> str:
        """优化：获取当前 Hires. fix 参数"""
        params = self.config["default_params"]
        upscale_factor = params.get("upscale_factor", "未设置")
        upscaler = params.get("upscaler", "未设置")
        denoising = params.get("denoising_strength", "未设置")
        hr_steps = params.get("hr_second_pass_steps", "未设置")

        return (
            f"- 放大倍数 (hr_scale): {upscale_factor}\n"
            f"- 上采样算法 (hr_upscaler): {upscaler}\n"
            f"- 重绘幅度 (denoising_strength): {denoising}\n"
            f"- 修复步数 (hr_second_pass_steps): {hr_steps}"
        )

    @command_group("绘图")
    def sd(self):
        pass

    @sd.command("检查")
    async def check(self, event: AstrMessageEvent):
        """服务状态检查"""
        try:
            webui_available, status = await self._check_webui_available()
            if webui_available:
                yield event.plain_result("✅ 同Webui连接正常")
            else:
                yield event.plain_result(f"❌ 同Webui无连接 (状态码: {status})，请检查配置和Webui工作状态")
        except Exception as e:
            logger.error(f"❌ 检查可用性错误，报错{e}")
            yield event.plain_result("❌ 检查可用性错误，请检查日志")

    @sd.command("刷新")
    async def refresh_cache(self, event: AstrMessageEvent):
        """清除资源缓存 (模型/采样器/VAE等)"""
        self.resource_cache = {}
        logger.info("SD 插件资源缓存已清除")
        yield event.plain_result("✅ 资源缓存已清除。下次列表查询将从 WebUI 重新获取。")

    @sd.command("画")
    async def handle_generate_image_command(self, event: AstrMessageEvent, prompt: str):
        """生成图像指令
        Args:
            prompt: 图像描述提示词
        """

        # --- 获取排队号 ---
        async with self.queue_lock:
            self.queue_counter += 1
            if self.queue_counter > 99999:  # 防止数字无限增大
                self.queue_counter = 1
            queue_num = self.queue_counter

        # --- 立即回复排队号 ---
        # (这会在等待并发信号量之前就发送给用户)
        try:
            yield event.plain_result(
                f"✅ 您已进入队列，排队号：【{queue_num}】\n"
                f"当前活跃任务: {self.active_tasks}/{self.max_concurrent_tasks}，请等待叫号。")
        except Exception:
            # 如果初始回复失败 (例如用户已离开)，则静默处理，但日志中应有记录
            logger.warning(f"队伍【{queue_num}】: 无法发送初始排队消息。")
            pass  # 无论如何都继续尝试生成


        async with self.task_semaphore:
            self.active_tasks += 1
            try:
                # 检查webui可用性
                if not (await self._check_webui_available())[0]:
                    yield event.plain_result(f"⚠️ 队伍【{queue_num}】: 同webui无连接，目前无法生成图片！")
                    return

                verbose = self.config["verbose"]
                if verbose:
                    yield event.plain_result(f"🖌️ 队伍【{queue_num}】: 开始生成图像，这可能需要一段时间...")

                # 生成提示词
                if self.config.get("enable_generate_prompt"):
                    generated_prompt = await self._generate_prompt(prompt)
                    logger.debug(f"队伍【{queue_num}】 LLM generated prompt: {generated_prompt}")
                    enable_positive_prompt_add_in_head_or_tail = self.config.get(
                        "enable_positive_prompt_add_in_head_or_tail", True)
                    if enable_positive_prompt_add_in_head_or_tail:
                        positive_prompt = self.config.get("positive_prompt_global", "") + generated_prompt
                    else:
                        positive_prompt = generated_prompt + self.config.get("positive_prompt_global", "")
                else:
                    enable_positive_prompt_add_in_head_or_tail = self.config.get(
                        "enable_positive_prompt_add_in_head_or_tail", True)
                    if enable_positive_prompt_add_in_head_or_tail:
                        positive_prompt = self.config.get("positive_prompt_global", "") + self._trans_prompt(prompt)
                    else:
                        positive_prompt = self._trans_prompt(prompt) + self.config.get("positive_prompt_global", "")

                # 输出正向提示词
                if self.config.get("enable_show_positive_prompt", False):
                    yield event.plain_result(f"队伍【{queue_num}】正向提示词：{positive_prompt}")

                # 生成图像 (Hires. fix 已包含在内)
                response = await self._call_t2i_api(positive_prompt)
                if not response.get("images"):
                    raise ValueError("API返回数据异常：生成图像失败")

                images = response["images"]

                # --- 发送图片前的叫号 ---
                yield event.plain_result(f"✅ 队伍【{queue_num}】的图片已生成：")

                if len(images) == 1:
                    # 直接将 API 返回的 base64 字符串传递给 Image.fromBase64
                    image_data_str = response["images"][0]
                    yield event.chain_result([Image.fromBase64(image_data_str)])
                else:
                    chain = []
                    for image_data_str in images:
                        # 直接将 API 返回的 base64 字符串传递给 Image.fromBase64
                        chain.append(Image.fromBase64(image_data_str))
                    yield event.chain_result(chain)

                if verbose:
                    yield event.plain_result(f"✅ 队伍【{queue_num}】: 图像发送完毕。")

            except ValueError as e:
                # 针对API返回异常的处理
                logger.error(f"队伍【{queue_num}】 API返回数据异常: {e}")
                yield event.plain_result(f"❌ 队伍【{queue_num}】图像生成失败: 参数异常，API调用失败")

            except ConnectionError as e:
                # 网络连接错误处理
                logger.error(f"队伍【{queue_num}】 网络连接失败: {e}")
                yield event.plain_result(f"⚠️ 队伍【{queue_num}】生成失败! 请检查网络连接和WebUI服务是否运行正常")

            except TimeoutError as e:
                # 处理超时错误
                logger.error(f"队伍【{queue_num}】 请求超时: {e}")
                yield event.plain_result(f"⚠️ 队伍【{queue_num}】请求超时，请稍后再试")

            except Exception as e:
                # 捕获所有其他异常
                logger.error(f"队伍【{queue_num}】 生成图像时发生其他错误: {e}")
                yield event.plain_result(f"❌ 队伍【{queue_num}】图像生成失败: 发生其他错误，请检查日志")
            finally:
                self.active_tasks -= 1

    @sd.command("详细")
    async def set_verbose(self, event: AstrMessageEvent):
        """切换详细输出模式（verbose）"""
        try:
            # 读取当前状态并取反
            current_verbose = self.config.get("verbose", True)
            new_verbose = not current_verbose

            # 更新配置
            self.config["verbose"] = new_verbose
            self.config.save_config()

            # 发送反馈消息
            status = "开启" if new_verbose else "关闭"
            yield event.plain_result(f"📢 详细输出模式已{status}")
        except Exception as e:
            logger.error(f"切换详细输出模式失败: {e}")
            yield event.plain_result("❌ 切换详细模式失败，请检查日志")

    @sd.command("高清")
    async def set_upscale(self, event: AstrMessageEvent):
        """(Hires. fix) 切换高分修复模式"""
        try:
            # 获取当前的 upscale 配置值
            current_upscale = self.config.get("enable_upscale", False)

            # 切换 enable_upscale 配置
            new_upscale = not current_upscale

            # 更新配置
            self.config["enable_upscale"] = new_upscale
            self.config.save_config()

            # 发送反馈消息
            status = "开启" if new_upscale else "关闭"
            yield event.plain_result(f"📢 Hires. fix (高分修复) 模式已{status}")

        except Exception as e:
            logger.error(f"切换 Hires. fix 模式失败: {e}")
            yield event.plain_result("❌ 切换 Hires. fix 模式失败，请检查日志")

    @sd.command("llm")
    async def set_generate_prompt(self, event: AstrMessageEvent):
        """切换生成提示词功能"""
        try:
            current_setting = self.config.get("enable_generate_prompt", False)
            new_setting = not current_setting
            self.config["enable_generate_prompt"] = new_setting
            self.config.save_config()

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 提示词生成功能已{status}")
        except Exception as e:
            logger.error(f"切换生成提示词功能失败: {e}")
            yield event.plain_result("❌ 切换生成提示词功能失败，请检查日志")

    @sd.command("提示词")
    async def set_show_prompt(self, event: AstrMessageEvent):
        """切换显示正向提示词功能"""
        try:
            current_setting = self.config.get("enable_show_positive_prompt", False)
            new_setting = not current_setting
            self.config["enable_show_positive_prompt"] = new_setting
            self.config.save_config()

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 显示正向提示词功能已{status}")
        except Exception as e:
            logger.error(f"切换显示正向提示词功能失败: {e}")
            yield event.plain_result("❌ 切换显示正向提示词功能失败，请检查日志")

    @sd.command("超时")
    async def set_timeout(self, event: AstrMessageEvent, time: int):
        """设置会话超时时间"""
        try:
            if time < 10 or time > 300:
                yield event.plain_result("⚠️ 超时时间需设置在 10 到 300 秒范围内")
                return

            self.config["session_timeout_time"] = time
            self.config.save_config()

            # 重新初始化 session
            self.session = None
            await self.ensure_session()

            yield event.plain_result(f"⏲️ 会话超时时间已设置为 {time} 秒")
        except Exception as e:
            logger.error(f"设置会话超时时间失败: {e}")
            yield event.plain_result("❌ 设置会话超时时间失败，请检查日志")

    @sd.command("配置")
    async def show_conf(self, event: AstrMessageEvent):
        """打印当前图像生成参数，包括当前使用的模型"""
        try:
            gen_params = self._get_generation_params()  # 获取当前图像参数
            scale_params = self._get_upscale_params()  # 获取图像增强参数
            prompt_guidelines = self.config.get("prompt_guidelines").strip() or "未设置"  # 获取提示词限制

            verbose = self.config.get("verbose", True)  # 获取详略模式
            upscale = self.config.get("enable_upscale", False)  # 图像增强模式
            show_positive_prompt = self.config.get("enable_show_positive_prompt", False)  # 是否显示正向提示词
            generate_prompt = self.config.get("enable_generate_prompt", False)  # 是否启用生成提示词

            conf_message = (
                f"⚙️  图像生成参数:\n{gen_params}\n\n"
                f"🔍  Hires. fix (高分修复) 参数:\n{scale_params}\n\n"
                f"🛠️  提示词附加要求: {prompt_guidelines}\n\n"
                f"📢  详细输出模式: {'开启' if verbose else '关闭'}\n\n"
                f"🔧  Hires. fix 模式: {'开启' if upscale else '关闭'}\n\n"
                f"📝  正向提示词显示: {'开启' if show_positive_prompt else '关闭'}\n\n"
                f"🤖  提示词生成模式: {'开启' if generate_prompt else '关闭'}"
            )

            yield event.plain_result(conf_message)
        except Exception as e:
            logger.error(f"获取生成参数失败: {e}")
            yield event.plain_result("❌ 获取图像生成参数失败，请检查配置是否正确")

    @sd.command("帮助")
    async def show_help(self, event: AstrMessageEvent):
        """(优化) 显示SDGenerator插件所有可用指令及其描述"""
        help_msg = [
            "🖼️ **绘图插件帮助指南**",
            "",
            "📜 **核心指令**:",
            "- `/绘图 画 [提示词]`：生成图片。 (示例: `/绘图 画 星空下的城堡`)",
            "- `/绘图 配置`：显示当前所有生效的配置参数。",
            "- `/绘图 检查`：检查 WebUI 的连接状态。",
            "- `/绘图 刷新`：清除插件的模型/VAE/采样器缓存 (添加新模型后使用)。",
            "- `/绘图 帮助`：显示本帮助信息。",
            "",
            "⚙️ **生成参数指令**:",
            "- `/绘图 尺寸 [宽度] [高度]`：设置基础分辨率 (1-2048)。",
            "- `/绘图 步数 [步数]`：设置采样步数 (10-50)。",
            "- `/绘图 种子 [数字]`：设置种子 (-1 为随机)。",
            "- `/绘图 clip [数字]`：设置 Clip Skip (推荐 1 或 2)。",
            "- `/绘图 批量 [数量]`：设置每轮生成的图片数量 (1-10)。",
            "- `/绘图 迭代 [次数]`：设置迭代次数 (1-5)。",
            "",
            "✨ **Hires. fix (高分修复) 指令**:",
            "- `/绘图 高清`：切换 Hires. fix (高分修复) 功能 [开启/关闭]。",
            "- `/绘图 h倍数 [倍数]`：设置 Hires. fix 放大倍数 (例如 1.5, 2)。",
            "- `/绘图 重绘 [幅度]`：设置 Hires. fix 重绘幅度 (0.0-1.0, 推荐 0.4)。",
            "- `/绘图 h步数 [步数]`：设置 Hires. fix 修复步数 (0-100, 0为自动)。",
            "- `/绘图 放大器 设置 [索引]`：设置 Hires. fix 使用的上采样算法。",
            "",
            "🎛️ **资源设置指令**:",
            "- `/绘图 模型 列表` / `设置 [索引]`：查看或切换基础模型。",
            "- `/绘图 vae 列表` / `设置 [索引]`：查看或切换 VAE。",
            "- `/绘图 采样器 列表` / `设置 [索引]`：查看或切换采样器。",
            "- `/绘图 放大器 列表`：查看可用的上采样算法。",
            "- `/绘图 lora`：(只读) 列出可用的 LoRA 模型。",
            "- `/绘图 embedding`：(只读) 显示已加载的 Embedding。",
            "",
            "🤖 **模式切换指令**:",
            "- `/绘图 llm`：切换 [LLM生成提示词 / 用户直出提示词] 模式。",
            "- `/绘图 详细`：切换 [详细输出 / 简洁输出] 模式。",
            "- `/绘图 提示词`：切换 [显示最终提示词 / 不显示] 模式。",
            "- `/绘图 超时 [秒数]`：设置连接超时时间 (10-300)。",
            "",
            "ℹ️ **注意事项**:",
            "- 提示词中的空格请用 `~` (波浪号) 替代, 或在配置中修改该字符。",
        ]
        yield event.plain_result("\n".join(help_msg))

    @sd.command("尺寸")
    async def set_resolution(self, event: AstrMessageEvent, width: int, height: int):
        """设置分辨率"""
        try:
            if not isinstance(height, int) or not isinstance(width,
                                                             int) or height < 1 or width < 1 or height > 2048 or width > 2048:
                yield event.plain_result("⚠️ 分辨率仅支持:1-2048之间的任意整数")
                return

            self.config["default_params"]["height"] = height
            self.config["default_params"]["width"] = width
            self.config.save_config()

            yield event.plain_result(f"✅ 图像生成的分辨率已设置为: 宽度——{width}，高度——{height}")
        except Exception as e:
            logger.error(f"设置分辨率失败: {e}")
            yield event.plain_result("❌ 设置分辨率失败，请检查日志")

    @sd.command("步数")
    async def set_step(self, event: AstrMessageEvent, step: int):
        """设置步数"""
        try:
            if step < 10 or step > 50:
                yield event.plain_result("⚠️ 步数需设置在 10 到 50 之间")
                return

            self.config["default_params"]["steps"] = step
            self.config.save_config()

            yield event.plain_result(f"✅ 步数已设置为: {step}")
        except Exception as e:
            logger.error(f"设置步数失败: {e}")
            yield event.plain_result("❌ 设置步数失败，请检查日志")

    # --- 新增命令 ---

    @sd.command("种子")
    async def set_seed(self, event: AstrMessageEvent, seed: int):
        """设置种子 (-1为随机)"""
        try:
            self.config["default_params"]["seed"] = int(seed)
            self.config.save_config()
            yield event.plain_result(f"✅ 种子已设置为: {seed}")
        except Exception as e:
            logger.error(f"设置种子失败: {e}")
            yield event.plain_result("❌ 设置种子失败，请检查日志")

    @sd.command("clip")
    async def set_clip_skip(self, event: AstrMessageEvent, skip: int):
        """设置 Clip Skip"""
        try:
            if skip < 1 or skip > 12:
                yield event.plain_result("⚠️ Clip Skip 建议设置在 1 到 12 之间 (通常为 1 或 2)")
                return
            self.config["default_params"]["clip_skip"] = skip
            self.config.save_config()
            yield event.plain_result(f"✅ Clip Skip 已设置为: {skip}")
        except Exception as e:
            logger.error(f"设置 Clip Skip 失败: {e}")
            yield event.plain_result("❌ 设置 Clip Skip 失败，请检查日志")

    @sd.command("重绘")
    async def set_denoising(self, event: AstrMessageEvent, strength: float):
        """设置 Hires. fix 的重绘幅度"""
        try:
            strength = float(strength)
            if not (0.0 <= strength <= 1.0):
                yield event.plain_result("⚠️ Hires. fix 重绘幅度必须在 0.0 到 1.0 之间")
                return
            self.config["default_params"]["denoising_strength"] = strength
            self.config.save_config()
            yield event.plain_result(f"✅ Hires. fix 重绘幅度已设置为: {strength}")
        except Exception as e:
            logger.error(f"设置重绘幅度失败: {e}")
            yield event.plain_result("❌ 设置重绘幅度失败，请输入有效的小数 (例如 0.4)")

    @sd.command("h步数")
    async def set_hr_steps(self, event: AstrMessageEvent, steps: int):
        """设置 Hires. fix 的修复步数"""
        try:
            if not (0 <= steps <= 100):
                yield event.plain_result("⚠️ Hires. fix 步数必须在 0 到 100 之间 (0为自动)")
                return
            self.config["default_params"]["hr_second_pass_steps"] = steps
            self.config.save_config()
            yield event.plain_result(f"✅ Hires. fix 修复步数已设置为: {steps}")
        except Exception as e:
            logger.error(f"设置 Hires. fix 步数失败: {e}")
            yield event.plain_result("❌ 设置 Hires. fix 步数失败，请检查日志")

    @sd.command("h倍数")
    async def set_hr_scale(self, event: AstrMessageEvent, scale: float):
        """设置 Hires. fix 的放大倍数"""
        try:
            scale = float(scale)
            if not (1.0 <= scale <= 8.0):
                yield event.plain_result("⚠️ Hires. fix 放大倍数必须在 1.0 到 8.0 之间")
                return
            self.config["default_params"]["upscale_factor"] = scale
            self.config.save_config()
            yield event.plain_result(f"✅ Hires. fix 放大倍数已设置为: {scale}x")
        except Exception as e:
            logger.error(f"设置 Hires. fix 放大倍数失败: {e}")
            yield event.plain_result("❌ 设置 Hires. fix 放大倍数失败，请输入有效的小数 (例如 1.5)")

    # --- 结束新增命令 ---

    @sd.command("批量")
    async def set_batch_size(self, event: AstrMessageEvent, batch_size: int):
        """设置批量生成的图片数量"""
        try:
            if batch_size < 1 or batch_size > 10:
                yield event.plain_result("⚠️ 图片生成的批数量需设置在 1 到 10 之间")
                return

            self.config["default_params"]["batch_size"] = batch_size
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成批数量已设置为: {batch_size}")
        except Exception as e:
            logger.error(f"设置批量生成数量失败: {e}")
            yield event.plain_result("❌ 设置图片生成批数量失败，请检查日志")

    @sd.command("迭代")
    async def set_n_iter(self, event: AstrMessageEvent, n_iter: int):
        """设置生成迭代次数"""
        try:
            if n_iter < 1 or n_iter > 5:
                yield event.plain_result("⚠️ 图片生成的迭代次数需设置在 1 到 5 之间")
                return

            self.config["default_params"]["n_iter"] = n_iter
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成的迭代次数已设置为: {n_iter}")
        except Exception as e:
            logger.error(f"设置生成迭代次数失败: {e}")
            yield event.plain_result("❌ 设置图片生成的迭代次数失败，请检查日志")

    @sd.group("模型")
    def model(self):
        pass

    @model.command("列表")
    async def list_model(self, event: AstrMessageEvent):
        """
        以“1. xxx.safetensors“形式打印可用的模型
        """
        try:
            models = await self._get_sd_model_list()  # 使用统一方法获取模型列表
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            model_list = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(models))
            yield event.plain_result(f"🖼️ 可用模型列表:\n{model_list}")

        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            yield event.plain_result("❌ 获取模型列表失败，请检查 WebUI 是否运行")

    @model.command("设置")
    async def set_base_model(self, event: AstrMessageEvent, model_index: int):
        """
        解析用户输入的索引，并设置对应的模型
        """
        try:
            models = await self._get_sd_model_list()
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            try:
                index = int(model_index) - 1  # 转换为 0-based 索引
                if index < 0 or index >= len(models):
                    yield event.plain_result("❌ 无效的模型索引，请使用 /sd model list 获取")
                    return

                selected_model = models[index]
                logger.debug(f"selected_model: {selected_model}")
                if await self._set_model(selected_model):
                    yield event.plain_result(f"✅ 模型已切换为: {selected_model}")
                else:
                    yield event.plain_result("⚠️ 切换模型失败，请检查 WebUI 状态")

            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")

        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            yield event.plain_result("❌ 切换模型失败，请检查日志")

    @sd.command("lora")
    async def list_lora(self, event: AstrMessageEvent):
        """
        列出可用的 LoRA 模型
        """
        try:
            lora_models = await self._get_lora_list()
            if not lora_models:
                yield event.plain_result("没有可用的 LoRA 模型。")
            else:
                lora_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(lora_models))
                yield event.plain_result(f"可用的 LoRA 模型:\n{lora_model_list}")
        except Exception as e:
            yield event.plain_result(f"获取 LoRA 模型列表失败: {str(e)}")

    @sd.group("采样器")
    def sampler(self):
        pass

    @sampler.command("列表")
    async def list_sampler(self, event: AstrMessageEvent):
        """
        列出所有可用的采样器
        """
        try:
            samplers = await self._get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            sampler_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(samplers))
            yield event.plain_result(f"🖌️ 可用采样器列表:\n{sampler_list}")
        except Exception as e:
            yield event.plain_result(f"获取采样器列表失败: {str(e)}")

    @sampler.command("设置")
    async def set_sampler(self, event: AstrMessageEvent, sampler_index: int):
        """
        设置采样器
        """
        try:
            samplers = await self._get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            try:
                index = int(sampler_index) - 1
                if index < 0 or index >= len(samplers):
                    yield event.plain_result("❌ 无效的采样器索引，请使用 /sd sampler list 获取")
                    return

                selected_sampler = samplers[index]
                self.config["default_params"]["sampler"] = selected_sampler
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置采样器为: {selected_sampler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置采样器失败: {str(e)}")

    @sd.group("放大器")
    def upscaler(self):
        pass

    @upscaler.command("列表")
    async def list_upscaler(self, event: AstrMessageEvent):
        """
        列出所有可用的上采样算法
        """
        try:
            upscalers = await self._get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            upscaler_list = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(upscalers))
            yield event.plain_result(f"🖌️ 可用上采样算法列表:\n{upscaler_list}")
        except Exception as e:
            yield event.plain_result(f"获取上采样算法列表失败: {str(e)}")

    @upscaler.command("设置")
    async def set_upscaler(self, event: AstrMessageEvent, upscaler_index: int):
        """
        设置上采样算法
        """
        try:
            upscalers = await self._get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            try:
                index = int(upscaler_index) - 1
                if index < 0 or index >= len(upscalers):
                    yield event.plain_result("❌ 无效的上采样算法索引，请检查 /sd upscaler list")
                    return

                selected_upscaler = upscalers[index]
                self.config["default_params"]["upscaler"] = selected_upscaler
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置上采样算法为: {selected_upscaler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置上采样算法失败: {str(e)}")

    # --- 新增 VAE 命令组 ---
    @sd.group("vae")
    def vae(self):
        pass

    @vae.command("列表")
    async def list_vae(self, event: AstrMessageEvent):
        """列出所有可用的 VAE"""
        try:
            vaes = await self._get_vae_list()
            if not vaes:
                yield event.plain_result("⚠️ 没有可用的 VAE (或 WebUI 无法访问)")
                return

            vae_list = "\n".join(f"{i + 1}. {v}" for i, v in enumerate(vaes))
            yield event.plain_result(f"🎨 可用 VAE 列表:\n{vae_list}")
        except Exception as e:
            yield event.plain_result(f"获取 VAE 列表失败: {str(e)}")

    @vae.command("设置")
    async def set_vae(self, event: AstrMessageEvent, vae_index: int):
        """根据索引设置 VAE (输入 0 设置为 Automatic)"""
        try:
            if int(vae_index) == 0:
                selected_vae = "Automatic"
            else:
                vaes = await self._get_vae_list()
                if not vaes:
                    yield event.plain_result("⚠️ 没有可用的 VAE")
                    return

                index = int(vae_index) - 1
                if index < 0 or index >= len(vaes):
                    yield event.plain_result("❌ 无效的 VAE 索引, 请使用 /sd vae list 获取")
                    return
                selected_vae = vaes[index]

            self.config["default_params"]["sd_vae"] = selected_vae
            self.config.save_config()
            yield event.plain_result(f"✅ 已设置 VAE 为: {selected_vae}")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字索引 (输入 0 可设为 Automatic)")
        except Exception as e:
            yield event.plain_result(f"设置 VAE 失败: {str(e)}")

    # --- 结束新增 VAE 命令组 ---

    @sd.command("embedding")
    async def list_embedding(self, event: AstrMessageEvent):
        """
        列出可用的 Embedding 模型
        """
        try:
            embedding_models = await self._get_embedding_list()
            if not embedding_models:
                yield event.plain_result("没有可用的 Embedding 模型。")
            else:
                embedding_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(embedding_models))
                yield event.plain_result(f"可用的 Embedding 模型:\n{embedding_model_list}")
        except Exception as e:
            yield event.plain_result(f"获取 Embedding 模型列表失败: {str(e)}")

    @llm_tool("generate_image")
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """Generate images using Stable Diffusion based on the given prompt.
        This function should only be called when the prompt contains keywords like "generate," "draw," or "create."
        It should not be mistakenly used for image searching.

        Args:
            prompt (string): The prompt or description used for generating images.
        """
        try:
            async for result in self.handle_generate_image_command(event, prompt):
                # 根据生成器的每一个结果返回响应
                yield result

        except Exception as e:
            logger.error(f"调用 generate_image 时出错: {e}")
            yield event.plain_result("❌ 图像生成失败，请检查日志")