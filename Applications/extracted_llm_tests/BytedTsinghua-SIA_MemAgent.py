# BytedTsinghua-SIA/MemAgent
# 5 LLM-backed test functions across 33 test files
# Source: https://github.com/BytedTsinghua-SIA/MemAgent

# --- tests/rollout/test_hf_rollout.py ---

def test_hf_rollout(n: int = 1, do_sample: bool = True, validate: bool = False):
    config = OmegaConf.create(BASE_HF_ROLLOUT_CONFIG)
    config.update({"n": n, "do_sample": do_sample})

    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    local_rank, rank, world_size = initialize_global_process_group()

    # Initialize model and tokenizer
    local_cache_path = "~/.cache/verl/rlhf"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "Qwen/Qwen2-7B-Instruct"
    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize FSDP model
    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model.to(torch.bfloat16)
    fsdp_model = prepare_fsdp_model(actor_model, world_size)

    # Initialize HFRollout and start generate
    hf_rollout = HFRollout(fsdp_model, OmegaConf.create(config))
    input = prepare_input_dataproto(tokenizer, config, validate).to(torch.cuda.current_device())
    outputs = hf_rollout.generate_sequences(input)

    # check generated batch size is expected
    generated_batch_size = outputs.batch.batch_size[0]
    assert generated_batch_size == input.batch.batch_size[0] * config.n

    for i in range(generated_batch_size):
        prompt_tokens = outputs.batch["prompts"][i]
        prompt_mask = prompt_tokens != tokenizer.pad_token_id
        prompt_tokens = prompt_tokens[prompt_mask]
        decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        response_tokens = outputs.batch["responses"][i]
        response_mask = response_tokens != tokenizer.pad_token_id
        response_tokens = response_tokens[response_mask]
        decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        attention_mask = outputs.batch["attention_mask"][i]
        position_ids = outputs.batch["position_ids"][i]
        prompt_length = outputs.batch["prompts"].size(1)
        response_length = outputs.batch["responses"].size(1)

        assert attention_mask.size(0) == prompt_length + response_length
        assert position_ids.size(0) == prompt_length + response_length

        # check response attention mask is expected
        response_attention = attention_mask[prompt_length:]
        eos_positions = (outputs.batch["responses"][i] == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_eos_pos = eos_positions[0].item()
            assert response_attention[: first_eos_pos + 1].all(), "Response attention mask should be 1 until EOS"
            if first_eos_pos + 1 < response_length:
                assert not response_attention[first_eos_pos + 1 :].any(), "Response attention mask should be 0 after EOS"
        else:
            assert response_attention.all(), "Response attention mask should be all 1 if no EOS token"

        # check response position ids is expected
        prompt_positions = position_ids[:prompt_length]
        response_positions = position_ids[prompt_length:]
        valid_response_length = min(len(response_tokens), response_length)
        if valid_response_length > 0:
            assert response_positions[0] == prompt_positions[-1] + 1
            for j in range(1, valid_response_length):
                assert response_positions[j] == response_positions[j - 1] + 1

        # print generated text for inspection
        if torch.distributed.get_rank() == 0:
            print(f"prompt: {decoded_prompt}")
            print(f"response: {decoded_response}")
            print("=" * 30)


# --- tests/rollout/test_sglang_spmd.py ---

def test_sglang_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    initialize_global_process_group()
    # fill rollout config
    max_prompt_length = 16
    max_response_length = 16

    # Initialize model and token
    local_cache_path = "~/.cache/verl/rlhf"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "Qwen/Qwen2-7B-Instruct"
    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")

    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path)
    actor_model.to(torch.bfloat16)

    sampling_params = dict(
        n=1,
        temperature=0,
        top_p=1,
        top_k=-1,
        max_new_tokens=max_response_length,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        ignore_eos=False,
    )

    tensor_parallel_size = 4
    device_mesh_kwargs = dict(mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=["dp", "tp", "pp"])
    inference_device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    print("building sglang rollout engine")
    llm = VerlEngine(
        model_path=local_model_path,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        device_mesh_cpu=inference_device_mesh_cpu["tp"],
        base_gpu_id=0,
        gpu_id_step=1,
    )

    llm.release_memory_occupation()
    print("start generation")
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    batch_size = input_ids.size(0)

    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_response_length,
        # max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        # renormalize_logits=True,
        output_scores=False,  # this is potentially very large
        return_dict_in_generate=True,
        use_cache=False,
    )  # may OOM when use_cache = True
    seq = output.sequences
    response = seq[:, max_prompt_length:]

    hf_response_tokens = tokenizer.batch_decode(response)
    print(f"hf response: {hf_response_tokens}")
    print(f"{sampling_params=}")
    idx_list = []
    batch_size = input_ids.shape[0]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for i in range(batch_size):
        idx_list.append(_pre_process_inputs(pad_token_id, input_ids[i]))

    outputs = llm.generate(input_ids=idx_list, sampling_params=sampling_params)
    sglang_response_tokens = []

    for output in outputs:
        print(f"{output=}")
        generated_text = output["text"]
        sglang_response_tokens.append(generated_text)

    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens), "Strings differ more than 10%:\n"
    print("Check Pass")


# --- tests/rollout/test_vllm_hf_loader.py ---

def test_vllm_with_hf():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."

    # fill rollout config
    max_prompt_length = 16
    max_response_length = 16

    # Initialize model and token
    local_cache_path = "~/.cache/verl/rlhf"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "deepseek-ai/deepseek-llm-7b-chat"
    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path)
    actor_model.to(torch.bfloat16)

    actor_model_config = AutoConfig.from_pretrained(local_model_path)

    temperature = 0
    top_p = 1

    kwargs = dict(n=1, temperature=temperature, top_p=top_p, max_tokens=max_response_length, logprobs=1, ignore_eos=True)

    if vllm_version in (
        "0.5.4",
        "0.6.3",
    ):
        kwargs["detokenize"] = False
    sampling_params = SamplingParams(**kwargs)

    tensor_parallel_size = 4

    llm = LLM(
        model=actor_model,
        tokenizer=tokenizer,
        model_hf_config=actor_model_config,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.1,
        load_format="hf",
    )

    print("start generation")
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    batch_size = input_ids.size(0)

    idx_list = []
    # parse idx from torch.Tensor to List[List[str]]
    for i in range(batch_size):
        idx_list.append(_pre_process_inputs(tokenizer.pad_token_id, input_ids[i]))
    outputs = llm.generate(prompt_token_ids=idx_list, sampling_params=sampling_params, use_tqdm=False)
    vllm_output = outputs[0].cuda()
    llm.free_cache_engine()
    llm = None
    import gc

    torch.cuda.empty_cache()
    gc.collect()

    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_response_length,
        # max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        # renormalize_logits=True,
        output_scores=False,  # this is potentially very large
        return_dict_in_generate=True,
        use_cache=False,
    )  # may OOM when use_cache = True
    seq = output.sequences
    response = seq[:, max_prompt_length:]

    hf_response_tokens = tokenizer.batch_decode(response)
    vllm_response_tokens = tokenizer.batch_decode(vllm_output)

    print(f"hf response: {hf_response_tokens}")
    print(f"vllm response: {vllm_response_tokens}")
    assert are_lists_similar(hf_response_tokens, vllm_response_tokens), "Strings differ more than 10%:\n"
    print("Check Pass")


# --- tests/rollout/test_vllm_multi_turn.py ---

def test_vllm_multi_turn():
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "Qwen/Qwen2-7B-Instruct"
    model_name = "/".join(model_path.split("/")[-2:])
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    # =========================== 1. Create hybrid ActorRollout workers ===========================
    # make openai client happy
    os.environ["no_proxy"] = ""
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout")
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    all_wg = {}
    wg_dicts = []
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
        wg_dicts.append(wg_dict)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    # =========================== 2. Create AsyncLLMServerManager  ===========================
    async_rollout_manager = AsyncLLMServerManager(
        config=config.actor_rollout_ref,
        worker_group=actor_rollout_wg,
    )

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()

    async_chat_scheduler = async_rollout_manager.chat_scheduler

    # =========================== 3. Multi turn rollout  ===========================
    async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
        assert exception is None, f"exception: {exception}"
        messages, round = info["messages"], info["round"]
        message = completions.choices[0].message
        messages.append({"role": message.role, "content": message.content})
        print(f"[round={round}] role: {message.role}, content: {message.content}")

        extra_headers = {"x-request-id": completions.id}
        if round == 0:
            messages.append({"role": "user", "content": "What is your name?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 1},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        elif round == 1:
            messages.append({"role": "user", "content": "What is your favorite color?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 2},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        else:
            print("Done!")

    messages = [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}]
    async_rollout_manager.submit_chat_completions(
        callback=callback,
        callback_additional_info={"messages": messages, "round": 0},
        model=model_name,
        messages=messages,
    )
    assert len(messages) == 6
    for round, message in enumerate(messages):
        if round % 2 == 0:
            assert message["role"] == "user"
        else:
            assert message["role"] == "assistant"

    # =========================== 4. Generate sequences  ===========================
    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",
            }
        ],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
        },
    )
    result = async_rollout_manager.generate_sequences(prompts=batch)
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert len(result) == 2
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len


# --- tests/rollout/test_vllm_spmd.py ---

def test_vllm_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    local_rank, rank, world_size = initialize_global_process_group()

    # Initialize model and token
    local_cache_path = "~/.cache/verl/rlhf"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "Qwen/Qwen2-7B-Instruct"
    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)

    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model.to(torch.bfloat16)

    # fill rollout config
    max_prompt_length = 16
    max_response_length = 32
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    print("start generation")
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    temperature = 0
    top_p = 1
    kwargs = dict(n=1, temperature=temperature, top_p=top_p, max_tokens=max_response_length, logprobs=1, ignore_eos=True)

    tensor_parallel_size = 4

    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )

    FSDP.set_state_dict_type(fsdp_model, state_dict_type=StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig())

    state_dict = fsdp_model.state_dict()

    sampling_params = SamplingParams(**kwargs)
    llm = LLM(
        model=local_model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        skip_tokenizer_init=False,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=1,
    )

    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    vllm_response_tokens = []
    for output in outputs:
        generated_text = output.outputs[0].text
        vllm_response_tokens.append(generated_text)

    world_size = torch.distributed.get_world_size()
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    model.load_weights(((name, param.full_tensor() if world_size != 1 else param) for name, param in state_dict.items()))

    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    verl_vllm_response_tokens = []
    for output in outputs:
        generated_text = output.outputs[0].text
        verl_vllm_response_tokens.append(generated_text)

    if torch.distributed.get_rank() == 0:
        print(f"vllm response: {vllm_response_tokens}")
        print(f"verl-vllm response: {verl_vllm_response_tokens}")
    assert are_lists_similar(vllm_response_tokens, verl_vllm_response_tokens), "Strings differ more than 10%:\n"
    print("Check Pass")
    torch.distributed.destroy_process_group()

