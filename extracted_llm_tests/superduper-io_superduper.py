# superduper-io/superduper
# 8 test functions with real LLM calls
# Source: https://github.com/superduper-io/superduper


# --- plugins/openai/plugin_test/test_model_openai.py ---

def test_embed():
    e = OpenAIEmbedding(identifier='text-embedding-ada-002')
    resp = e.predict('Hello, world!')

    assert str(resp.dtype) == 'float32'

def test_batch_embed():
    e = OpenAIEmbedding(identifier='text-embedding-ada-002', batch_size=1)
    resp = e.predict_batches(['Hello', 'world!'])

    assert len(resp) == 2
    assert all(str(x.dtype) == 'float32' for x in resp)

def test_chat():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo', prompt='Hello, {context}')
    resp = e.predict('', context=['world!'])

    assert isinstance(resp, str)

def test_batch_chat():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo')
    resp = e.predict_batches([(('Hello, world!',), {})])

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


# --- plugins/transformers/plugin_test/test_llm_training_skip.py ---

def test_full_finetune(db):

    trainer = get_trainer(db)
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    trainer.use_lora = False
    # Don't log to db if full finetune cause the large files
    trainer.log_to_db = False
    output_dir = os.path.join(save_folder, "test_full_finetune")
    trainer.output_dir = output_dir

    db.apply(llm)

    llm = LLM(
        identifier="llm",
        model_name_or_path=transformers.trainer.get_last_checkpoint(output_dir),
        model_kwargs=dict(device_map="auto"),
    )

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0

def test_lora_finetune(db):
    trainer = get_trainer(db)
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    output_dir = os.path.join(save_folder, "test_lora_finetune")
    trainer.output_dir = output_dir

    db.apply(llm)

    llm = db.load("model", "llm-finetune")

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0

def test_qlora_finetune(db):

    trainer = get_trainer(db)

    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    trainer.bits = 4
    output_dir = os.path.join(save_folder, "test_qlora_finetune")
    trainer.output_dir = output_dir

    db.apply(llm)

    llm = db.load("model", "llm-finetune")

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0

def test_local_ray_deepspeed_lora_finetune(db):

    trainer = get_trainer(db)

    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    deepspeed = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 0,
        },
    }

    trainer.use_lora = True
    output_dir = os.path.join(save_folder, "test_local_ray_deepspeed_lora_finetune")
    trainer.output_dir = output_dir
    trainer.deepspeed = deepspeed
    trainer.bits = 4

    db.apply(llm)

    llm = db.load("model", "llm-finetune")

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0

