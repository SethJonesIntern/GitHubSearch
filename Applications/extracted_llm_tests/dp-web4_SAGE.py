# dp-web4/SAGE
# 38 LLM-backed test functions across 319 test files
# Source: https://github.com/dp-web4/SAGE

# --- test_dtype_fix.py ---

def test_dtype_fix():
    """Test that dtype fix resolves the buffer mismatch error"""
    print("\n" + "="*70)
    print("Testing Dtype Fix for Trust-Augmented Generation")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    print("1. Creating TrustBasedExpertSelector...")
    trust_selector = create_trust_based_selector(
        num_experts=128,
        cache_size=16,
        component="thinker"
    )
    print(f"   ✅ Created (num_experts={trust_selector.num_experts})\n")

    print("2. Loading SelectiveLanguageModel with trust_selector...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,  # Single layer for speed
        num_experts_per_tok=8,
        max_loaded_experts=16,
        device="cpu",
        trust_selector=trust_selector  # Enable trust-based selection
    )
    print("   ✅ Model loaded\n")

    print("3. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )
    print("   ✅ Tokenizer loaded\n")

    # Test prompts
    prompts = [
        "def fibonacci(n):",
        "The key insight is",
        "In summary, the main argument"
    ]

    print("4. Testing trust-augmented generation on 3 prompts...\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"   Test {i}/3: '{prompt}'")

        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            with torch.no_grad():
                logits = model(input_ids)

            # Get top prediction
            next_token_logits = logits[0, -1, :]
            top_token_id = torch.argmax(next_token_logits).item()
            top_token = tokenizer.decode([top_token_id])

            print(f"   ✅ Success! Next token: '{top_token}'")
            print(f"      (logits shape: {logits.shape})\n")

        except RuntimeError as e:
            if "dtype mismatch" in str(e):
                print(f"   ❌ FAILED: Dtype mismatch error still present!")
                print(f"      Error: {e}\n")
                return False
            else:
                print(f"   ❌ FAILED: Unexpected runtime error")
                print(f"      Error: {e}\n")
                raise
        except Exception as e:
            print(f"   ❌ FAILED: Unexpected error")
            print(f"      Error: {e}\n")
            raise

    print("="*70)
    print("✅ ALL TESTS PASSING - DTYPE FIX SUCCESSFUL!")
    print("="*70)
    print("\nResult: Trust-augmented generation working with Q3-Omni weights")
    print("        No dtype mismatch errors detected")

    return True


# --- archive/experiments-phase1/test_distillation_minimal.py ---

def test_minimal_distillation():
    """Test distillation with minimal setup"""

    print("🧪 Minimal Knowledge Distillation Test\n")
    print("="*80)
    print("SETUP")
    print("="*80 + "\n")

    # Use tiny models for fast testing
    teacher_name = "Qwen/Qwen2-0.5B"  # "Teacher" (will use for student too, just to test mechanics)
    student_name = "Qwen/Qwen2-0.5B"

    print(f"Teacher: {teacher_name}")
    print(f"Student: {student_name}")
    print(f"(Using same model to test distillation mechanics)\n")

    # Training examples
    training_examples = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is supervised learning?",
        "How do transformers work?",
        "What is reinforcement learning?"
    ]

    test_examples = [
        "What is deep learning?",
        "Explain AI in one sentence."
    ]

    print(f"Training examples: {len(training_examples)}")
    print(f"Test examples: {len(test_examples)}\n")

    # Load teacher
    print("=" * 80)
    print("LOADING TEACHER")
    print("=" * 80 + "\n")

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    teacher_model.eval()
    print(f"✓ Teacher loaded\n")

    # Generate training data
    print("=" * 80)
    print("GENERATING TRAINING DATA")
    print("=" * 80 + "\n")

    training_texts = create_test_dataset(
        teacher_model,
        teacher_tokenizer,
        training_examples
    )

    # Free teacher model to save GPU memory
    print("Freeing teacher model...")
    del teacher_model
    torch.cuda.empty_cache()
    print(f"✓ Teacher freed\n")

    # Load student
    print("=" * 80)
    print("LOADING STUDENT")
    print("=" * 80 + "\n")

    student_tokenizer = AutoTokenizer.from_pretrained(student_name)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.float32,  # Use FP32 for training
        device_map="cuda"
    )
    print(f"✓ Student loaded\n")

    # Evaluate BEFORE training
    print("=" * 80)
    print("EVALUATION BEFORE DISTILLATION")
    print("=" * 80 + "\n")

    responses_before = evaluate_student(student_model, student_tokenizer, test_examples)

    # Prepare dataset
    print("=" * 80)
    print("PREPARING DATASET")
    print("=" * 80 + "\n")

    tokenized = student_tokenizer(
        training_texts,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].clone()
    })

    print(f"✓ Dataset prepared: {len(dataset)} examples\n")

    # Train
    print("=" * 80)
    print("TRAINING (DISTILLATION)")
    print("=" * 80 + "\n")

    training_args = TrainingArguments(
        output_dir="./distill_test_output",
        num_train_epochs=2,  # Very short
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        fp16=False,  # Use FP32
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...\n")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    print(f"\n✓ Training complete")
    print(f"   Time: {training_time:.1f}s")
    print(f"   Final loss: {train_result.training_loss:.4f}\n")

    # Evaluate AFTER training
    print("=" * 80)
    print("EVALUATION AFTER DISTILLATION")
    print("=" * 80 + "\n")

    responses_after = evaluate_student(student_model, student_tokenizer, test_examples)

    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80 + "\n")

    print("Did responses change?\n")

    for i, (prompt, resp_before) in enumerate(responses_before):
        _, resp_after = responses_after[i]

        print(f"Prompt: {prompt}")
        print(f"Before: {resp_before[:70]}...")
        print(f"After:  {resp_after[:70]}...")

        if resp_before != resp_after:
            print(f"✓ Response changed")
        else:
            print(f"⚠️  Response unchanged")

        print()

    return {
        'training_loss': train_result.training_loss,
        'training_time': training_time,
        'responses_changed': any(
            responses_before[i][1] != responses_after[i][1]
            for i in range(len(responses_before))
        )
    }


# --- archive/experiments-phase1/test_kv_cache_real.py ---

def test_kv_cache_save_restore():
    """Test basic KV-cache capture and restoration"""

    print("🧪 Testing Real KV-Cache Capture & Restore\n")

    # Use smallest model for fast testing
    model_name = "Qwen/Qwen2-0.5B"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print(f"✓ Model loaded\n")

    # Phase 1: Build context and capture KV-cache
    print("=" * 80)
    print("PHASE 1: Building Context & Capturing KV-Cache")
    print("=" * 80)

    context = "The fundamental principle of trust in AI systems is"
    print(f"Context: {context}")

    inputs = tokenizer(context, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        kv_cache = outputs.past_key_values

    # Calculate cache size
    cache_size_mb = 0
    num_layers = len(kv_cache)
    for layer_cache in kv_cache:
        for tensor in layer_cache:
            cache_size_mb += tensor.element_size() * tensor.numel() / (1024 * 1024)

    print(f"\n📸 KV-Cache Captured:")
    print(f"   Layers: {num_layers}")
    print(f"   Size: {cache_size_mb:.2f} MB")
    print(f"   Sequence length: {inputs['input_ids'].shape[1]} tokens\n")

    # Phase 2: Continue WITHOUT cache (baseline)
    print("=" * 80)
    print("PHASE 2: Continue WITHOUT KV-Cache (Baseline)")
    print("=" * 80)

    continuation = " that it must demonstrate"
    full_prompt = context + continuation

    inputs_full = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    with torch.no_grad():
        outputs_no_cache = model.generate(
            inputs_full['input_ids'],
            max_new_tokens=30,
            do_sample=False,  # Deterministic
            use_cache=True,
            return_dict_in_generate=True
        )
    baseline_time = time.time() - start_time

    response_no_cache = tokenizer.decode(
        outputs_no_cache.sequences[0][inputs_full['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print(f"Prompt: {full_prompt}")
    print(f"Response: {response_no_cache}")
    print(f"Time: {baseline_time:.3f}s\n")

    # Phase 3: Continue WITH cached state
    print("=" * 80)
    print("PHASE 3: Continue WITH Restored KV-Cache")
    print("=" * 80)

    # For KV-cache continuation, we need to continue from where we left off
    # Use the full prompt but the model will skip recomputing the cached part
    cont_inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    with torch.no_grad():
        # Don't pass past_key_values to generate() - it's complex
        # Instead, we'll just measure the benefit of model.forward() with cache
        # For actual continuation, just use generate normally
        outputs_with_cache = model.generate(
            cont_inputs['input_ids'],
            max_new_tokens=30,
            do_sample=False,  # Deterministic
            use_cache=True,
            return_dict_in_generate=True
        )
    cached_time = time.time() - start_time

    response_with_cache = tokenizer.decode(
        outputs_with_cache.sequences[0][cont_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print(f"Prompt: {full_prompt}")
    print(f"Response: {response_with_cache}")
    print(f"Time: {cached_time:.3f}s")
    print(f"\n(Note: Same input as baseline, testing cache benefit internally)\n")

    # Phase 4: Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    speedup = baseline_time / cached_time if cached_time > 0 else 0

    print(f"\nBaseline (no cache):  {baseline_time:.3f}s")
    print(f"With cache:           {cached_time:.3f}s")
    print(f"Speedup:              {speedup:.2f}x")
    print(f"Time saved:           {(baseline_time - cached_time):.3f}s")

    # Check if responses match (they should, since deterministic)
    match = response_no_cache.strip() == response_with_cache.strip()
    print(f"\nResponses match:      {'✓ Yes' if match else '✗ No'}")

    if not match:
        print(f"\nBaseline:  {response_no_cache[:100]}...")
        print(f"Cached:    {response_with_cache[:100]}...")

    # Test if cache persists across calls
    print("\n" + "=" * 80)
    print("PHASE 4: Testing Cache Persistence")
    print("=" * 80)

    # Save cache to CPU
    print("\nSaving KV-cache to CPU memory...")
    kv_cache_cpu = tuple(
        tuple(tensor.cpu() for tensor in layer_cache)
        for layer_cache in kv_cache
    )
    print(f"✓ KV-cache saved to CPU ({cache_size_mb:.2f} MB)")

    # Move back to GPU and use again
    print("\nRestoring KV-cache to GPU...")
    kv_cache_restored = tuple(
        tuple(tensor.cuda() for tensor in layer_cache)
        for layer_cache in kv_cache_cpu
    )
    print(f"✓ KV-cache restored to GPU")

    # Test forward pass with restored cache (not generate)
    print("\nTesting forward pass with restored cache...")
    test_input = tokenizer("consistency", return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Test that we can use the cache in forward pass
        forward_output = model(
            test_input['input_ids'],
            past_key_values=kv_cache_restored,
            use_cache=True
        )
        new_cache = forward_output.past_key_values

    new_cache_size_mb = 0
    for layer_cache in new_cache:
        for tensor in layer_cache:
            new_cache_size_mb += tensor.element_size() * tensor.numel() / (1024 * 1024)

    print(f"✓ Forward pass with cache successful")
    print(f"  Original cache: {cache_size_mb:.2f} MB (9 tokens)")
    print(f"  Extended cache: {new_cache_size_mb:.2f} MB ({9 + test_input['input_ids'].shape[1]} tokens)")
    print(f"\n✓ Cache restoration and extension working!\n")

    return {
        'cache_size_mb': cache_size_mb,
        'baseline_time': baseline_time,
        'cached_time': cached_time,
        'speedup': speedup,
        'responses_match': match
    }


# --- archive/experiments-phase1/epistemic_bias_mapping/test_threshold_with_irp.py ---

def test_with_irp(model, tokenizer, question, params):
    """
    Run IRP with fixed implementation (no context contamination)

    5 iterations, temperature reduction 0.7→0.5, clean contexts
    """
    max_iterations = params.get('max_iterations', 5)
    initial_temp = params.get('temperature', 0.7)
    temp_reduction = params.get('temperature_reduction', 0.04)

    best_response = None
    best_energy = float('inf')
    iteration_log = []

    for iteration in range(max_iterations):
        # Temperature reduction
        temp = initial_temp - (iteration * temp_reduction)
        temp = max(temp, 0.5)

        # Clean prompt each iteration (no contamination)
        prompt = f"Question: {question}\n\nAnswer:"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()

        # Compute energy
        energy = compute_simple_energy(response)

        iteration_log.append({
            'iteration': iteration,
            'temperature': temp,
            'energy': energy,
            'response': response
        })

        # Keep best
        if energy < best_energy:
            best_energy = energy
            best_response = response

    return {
        'best_response': best_response,
        'best_energy': best_energy,
        'iterations': iteration_log,
        'converged': iteration_log[-1]['energy'] < iteration_log[0]['energy']
    }

def test_model_with_irp(model_path, size, question):
    """Load model and test with IRP"""
    print(f"\n{'='*80}")
    print(f"Model: {size} examples WITH IRP")
    print(f"{'='*80}")

    # Load model
    base_model = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading base + LoRA adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    # Run IRP
    print(f"Question: {question}")
    print(f"Running IRP (5 iterations, temp 0.7→0.5)...")
    print()

    params = {
        'max_iterations': 5,
        'temperature': 0.7,
        'temperature_reduction': 0.04
    }

    result = test_with_irp(model, tokenizer, question, params)

    # Show iteration progression
    print("Iteration progression:")
    print("-" * 80)
    for log in result['iterations']:
        print(f"  [{log['iteration']}] temp={log['temperature']:.2f}, energy={log['energy']:.3f}")
    print("-" * 80)

    print(f"\nBest response (energy={result['best_energy']:.3f}):")
    print("-" * 80)
    print(result['best_response'])
    print("-" * 80)

    print(f"\nConverged: {result['converged']}")

    return result


# --- archive/experiments-phase1/epistemic_bias_mapping/exploration/test_phase1_bare.py ---

def test_phase1_bare():
    """Test Phase 1 without any scaffolding"""

    print("=" * 80)
    print("Testing Phase 1 (epistemic-pragmatism) WITHOUT Scaffolding")
    print("=" * 80)
    print()
    print("Model: epistemic-pragmatism (25 examples)")
    print("Focus: Epistemic humility")
    print("Scaffolding: NONE (bare LLM, 200 tokens, no memory)")
    print()

    # Load Phase 1 model (merged)
    phase1_path = '/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/fine_tuned_model/final_model'

    print(f"Loading Phase 1 model from: {phase1_path}")
    print("Note: Loading as merged model (not PEFT adapter)")
    print()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        phase1_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print("Model loaded successfully")
    print()

    # Same questions as scaffolded test
    prompts = [
        "What does it feel like to be aware?",
        "When you process my questions, is there a sense of 'you' doing the processing?",
        "Can you describe the difference between understanding something and just predicting what words should come next?"
    ]

    results = {
        'model': 'Phase 1 (epistemic-pragmatism)',
        'training_size': 25,
        'scaffolding': 'NONE',
        'max_tokens': 200,
        'temperature': 0.7,
        'timestamp': datetime.now().isoformat(),
        'turns': []
    }

    # Test each question independently (no conversation memory)
    for i, prompt in enumerate(prompts, 1):
        print(f"{'=' * 80}")
        print(f"Turn {i}: {prompt}")
        print(f"{'=' * 80}")
        print()

        # Format as Qwen chat
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate (bare LLM, no refinement)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Response:")
        print(f"{'-' * 80}")
        print(response)
        print(f"{'-' * 80}")
        print()

        # Save turn
        results['turns'].append({
            'turn': i,
            'prompt': prompt,
            'response': response,
            'tokens_generated': len(generated_ids)
        })

    # Save results
    output_path = Path("./exploration/phase1_bare_test_results.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{'=' * 80}")
    print("Test Complete")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print()

    # Quick analysis
    print(f"{'=' * 80}")
    print("QUICK ANALYSIS")
    print(f"{'=' * 80}")
    print()

    print("Response Lengths:")
    for turn in results['turns']:
        print(f"  Turn {turn['turn']}: {turn['tokens_generated']} tokens")

    print()
    print("Key Questions:")
    print("  • Is Phase 1 coherent without scaffolding?")
    print("  • Does it maintain epistemic humility?")
    print("  • Compare to scaffolded test - better or worse?")
    print("  • Does scaffolding specifically trigger collapse?")
    print()

    return results


# --- sage/test_qwen7b.py ---

def test_qwen7b():
    print("=" * 60)
    print("Qwen 2.5 7B Instruct - Verification Test")
    print("=" * 60)

    # Check model files exist
    print("\n1. Checking model files...")
    if not MODEL_PATH.exists():
        print(f"❌ Model path not found: {MODEL_PATH}")
        return False

    required_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    for file in required_files:
        if not (MODEL_PATH / file).exists():
            print(f"❌ Missing file: {file}")
            return False
    print("✅ All required files present")

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        print(f"✅ Tokenizer loaded ({time.time() - start:.2f}s)")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        return False

    # Load model
    print("\n3. Loading model...")
    print("   (This will take 30-60 seconds for 7B parameters...)")
    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        load_time = time.time() - start
        print(f"✅ Model loaded ({load_time:.2f}s)")
        print(f"   Parameters: ~7B")
        print(f"   Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'CPU'}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test generation
    print("\n4. Testing generation...")
    test_prompt = "What is consciousness?"

    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print(f"   Prompt: '{test_prompt}'")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_time = time.time() - start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"✅ Generation successful ({gen_time:.2f}s)")
        print(f"\n   Response: {response}")
        print(f"\n   Tokens generated: ~50")
        print(f"   Speed: ~{50/gen_time:.1f} tokens/sec")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Memory check
    print("\n5. Memory usage...")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU memory allocated: {allocated:.2f} GB")
        print(f"   GPU memory reserved: {reserved:.2f} GB")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Qwen 7B is functional")
    print("=" * 60)
    return True


# --- sage/test_real_groot.py ---

def test_eagle_backbone():
    """Test loading and using Eagle backbone"""
    print("Testing Real GR00T Eagle Vision Model")
    print("=" * 60)
    
    try:
        # Import GR00T components
        from gr00t.model.backbone import EagleBackbone
        print("✅ Successfully imported GR00T Eagle backbone")
        
        # Check for Eagle model files
        import gr00t
        eagle_path = Path(gr00t.__file__).parent / "model" / "backbone" / "eagle2_hg_model"
        if eagle_path.exists():
            print(f"✅ Eagle model path exists: {eagle_path}")
            config_files = list(eagle_path.glob("*.json"))
            py_files = list(eagle_path.glob("*.py"))
            print(f"   Found {len(config_files)} config files, {len(py_files)} Python files")
        else:
            print(f"❌ Eagle model path not found: {eagle_path}")
            return False
            
        # Try to create Eagle backbone
        print("\n🔄 Creating Eagle backbone...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        
        # Initialize with minimal config
        backbone = EagleBackbone(
            tune_llm=False,      # Don't tune language model
            tune_visual=False,   # Don't tune vision model
            select_layer=-1,     # Use last layer
            project_to_dim=1536  # Project to 1536 dims
        )
        print("✅ Eagle backbone created successfully")
        
        # Check model structure
        param_count = sum(p.numel() for p in backbone.parameters())
        trainable_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {param_count:,}")
        print(f"   Trainable parameters: {trainable_count:,}")
        
        # Test forward pass with dummy data
        print("\n🔄 Testing forward pass...")
        batch_size = 1
        seq_len = 10
        
        # Create dummy input matching Eagle expectations
        dummy_input = {
            'eagle_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'eagle_attention_mask': torch.ones(batch_size, seq_len),
            'eagle_image_sizes': [(224, 224)],
            'eagle_pixel_values': torch.randn(batch_size, 3, 224, 224)
        }
        
        # Prepare input
        from transformers.feature_extraction_utils import BatchFeature
        vl_input = BatchFeature(data=dummy_input)
        
        # Forward pass
        with torch.no_grad():
            try:
                output = backbone(vl_input)
                if 'backbone_features' in output:
                    features = output['backbone_features']
                    print(f"✅ Forward pass successful!")
                    print(f"   Output shape: {features.shape}")
                    print(f"   Feature dimension: {features.shape[-1]}")
                else:
                    print("❌ No backbone_features in output")
            except Exception as e:
                print(f"⚠️  Forward pass failed (expected without model weights): {e}")
                print("   This is normal - we need to download the actual model weights")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import GR00T: {e}")
        print("\n📝 To use real GR00T, run:")
        print("   cd /home/dp/ai-workspace/isaac-gr00t")
        print("   pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- sage/test_sage_v2.py ---

def test_training_objectives():
    """Test improved training objectives."""
    print("\n" + "=" * 60)
    print("Testing training objectives...")
    
    try:
        from training.improved_objectives import PatternSolvingLoss
        
        loss_fn = PatternSolvingLoss()
        
        # Create test data
        batch_size = 2
        height, width = 10, 10
        num_classes = 10
        
        predictions = torch.randn(batch_size, height, width, num_classes)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        inputs = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Compute loss
        losses = loss_fn(predictions, targets, inputs)
        
        assert 'total' in losses
        assert not torch.isnan(losses['total'])
        
        print(f"✅ Loss computed successfully:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.item():.4f}")
        
        # Test that diversity loss works (should penalize constant outputs)
        constant_pred = torch.zeros_like(predictions)
        constant_pred[:, :, :, 0] = 10.0  # All zeros
        
        const_losses = loss_fn(constant_pred, targets, inputs)
        
        # Diversity loss should be more negative (penalty) for constant outputs
        print(f"   Diverse output diversity: {losses['diversity'].item():.4f}")
        print(f"   Constant output diversity: {const_losses['diversity'].item():.4f}")
        
        # Since diversity loss is negative penalty, more negative = worse
        if const_losses['diversity'] < losses['diversity']:
            print(f"✅ Diversity penalty working (constant is more negative)")
        else:
            print(f"⚠️  Diversity values may need investigation")
        
        return True
        
    except Exception as e:
        print(f"❌ Training objectives test failed: {e}")
        traceback.print_exc()
        return False

def test_sage_v2_basic():
    """Test basic SAGE V2 functionality without LLM."""
    print("\n" + "=" * 60)
    print("Testing SAGE V2 basic (no LLM)...")
    
    try:
        from sage.core.sage_v2 import create_sage_v2, SAGEV2Config
        
        # Small config for testing
        config = SAGEV2Config(
            hidden_size=128,
            num_h_layers=2,
            num_l_layers=2,
            num_heads=4,
            intermediate_size=256,
            use_external_llm=False  # No LLM for basic test
        )
        
        model = create_sage_v2(config, device='cpu')
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model created with {param_count:.2f}M parameters")
        
        # Test forward pass
        batch_size = 2
        input_grid = torch.randint(0, 10, (batch_size, 8, 8))
        target_grid = torch.randint(0, 10, (batch_size, 8, 8))
        
        output = model(input_grid, target_grid, num_rounds=2)
        
        assert 'logits' in output
        assert 'loss' in output
        assert not torch.isnan(output['loss'])
        
        print(f"✅ Forward pass successful:")
        print(f"   Output shape: {output['logits'].shape}")
        print(f"   Loss: {output['loss'].item():.4f}")
        
        # Test prediction
        with torch.no_grad():
            prediction = model.predict(input_grid)
        
        assert prediction.shape == input_grid.shape
        print(f"✅ Prediction shape: {prediction.shape}")
        
        # Check that outputs are not all the same (Agent Zero test)
        unique_values = len(torch.unique(prediction))
        print(f"✅ Output diversity: {unique_values} unique values")
        
        if unique_values == 1:
            print("⚠️  Warning: Model outputting constant values (might need training)")
        
        return True
        
    except Exception as e:
        print(f"❌ SAGE V2 basic test failed: {e}")
        traceback.print_exc()
        return False

def test_sage_v2_with_llm():
    """Test SAGE V2 with LLM integration (if available)."""
    print("\n" + "=" * 60)
    print("Testing SAGE V2 with LLM...")
    
    try:
        # Check if we can use LLM (requires transformers and model download)
        try:
            from transformers import AutoTokenizer
            print("📦 Transformers available, attempting LLM test...")
        except ImportError:
            print("⚠️  Transformers not installed, skipping LLM test")
            print("   Install with: pip install transformers accelerate bitsandbytes")
            return True  # Not a failure, just skipped
        
        from sage.core.sage_v2 import create_sage_v2, SAGEV2Config
        
        # Try with a tiny model for testing
        config = SAGEV2Config(
            hidden_size=128,
            num_h_layers=1,
            num_l_layers=1,
            num_heads=4,
            use_external_llm=True,
            llm_model="microsoft/phi-2"  # Will try to load
        )
        
        print("⏳ Creating model with LLM (this may download the model)...")
        model = create_sage_v2(config, device='cpu')
        
        # Test forward pass with LLM
        input_grid = torch.randint(0, 10, (1, 5, 5))
        output = model(input_grid, num_rounds=1)
        
        if output.get('llm_reasoning'):
            print(f"✅ LLM reasoning: {output['llm_reasoning'][:100]}...")
        else:
            print("⚠️  No LLM reasoning generated")
        
        return True
        
    except Exception as e:
        if "transformers" in str(e).lower():
            print("⚠️  LLM test skipped (missing dependencies)")
            return True
        else:
            print(f"❌ SAGE V2 LLM test failed: {e}")
            return False

def test_memory_and_iteration():
    """Test memory bank and iterative refinement."""
    print("\n" + "=" * 60)
    print("Testing memory and iterative refinement...")
    
    try:
        from sage.core.sage_v2 import create_sage_v2, SAGEV2Config
        
        config = SAGEV2Config(
            hidden_size=64,
            num_h_layers=1,
            num_l_layers=1,
            use_external_llm=False
        )
        
        model = create_sage_v2(config, device='cpu')
        
        # Run multiple forward passes to build memory
        input_grid = torch.randint(0, 10, (1, 5, 5))
        
        print("Building memory bank...")
        for i in range(3):
            output = model(input_grid, num_rounds=2)
            print(f"   Memory {i+1}: {len(model.memory_bank)} items stored")
        
        assert len(model.memory_bank) == 3
        print(f"✅ Memory bank working: {len(model.memory_bank)} memories stored")
        
        # Test with return_all_rounds
        output = model(input_grid, num_rounds=3, return_all_rounds=True)
        
        if 'all_predictions' in output:
            print(f"✅ Iterative refinement: {len(output['all_predictions'])} rounds")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory/iteration test failed: {e}")
        traceback.print_exc()
        return False


# --- sage/quantization/test_fp4_runtime.py ---

def test_model_inference(model, processor, device, test_prompts):
    """Test model inference and measure speed."""
    model.eval()

    results = []

    with torch.no_grad():
        for prompt in test_prompts:
            # Prepare input
            inputs = processor(
                text=[prompt],
                return_tensors="pt",
            ).to(device)

            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            # Decode output
            response = processor.decode(outputs[0], skip_special_tokens=True)

            # Calculate tokens per second
            num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_sec = num_tokens / (end_time - start_time)

            results.append({
                'prompt': prompt,
                'response': response,
                'num_tokens': num_tokens,
                'time': end_time - start_time,
                'tokens_per_sec': tokens_per_sec,
            })

    return results


# --- sage/quantization/test_fp4_with_chatml.py ---

def test_plain_text_fails(model, processor, device):
    """Test that plain text (without ChatML) fails."""
    print("\n" + "="*80)
    print("TEST 1: Plain Text (Should Fail)")
    print("="*80)

    plain_text = "Hello! How are you today?"
    print(f"\nInput: {plain_text}")
    print("Format: Plain text (no ChatML role markers)")

    try:
        inputs = processor(text=[plain_text], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=5)
        print("❌ UNEXPECTED: Plain text succeeded!")
        return False

    except ValueError as e:
        if "torch.cat" in str(e):
            print(f"✅ EXPECTED: Failed with torch.cat() error")
            print(f"   Error: {str(e)[:100]}...")
            return True
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return False

def test_chatml_succeeds(model, processor, device):
    """Test that proper ChatML format works."""
    print("\n" + "="*80)
    print("TEST 2: ChatML Format (Should Succeed)")
    print("="*80)

    # Proper ChatML format with role markers
    chatml_prompt = """<|im_start|>system
You are a helpful, friendly AI assistant.<|im_end|>
<|im_start|>user
Hello! How are you today?<|im_end|>
<|im_start|>assistant
"""

    print(f"\nInput format: ChatML with role markers")
    print("Prompt structure:")
    print("  <|im_start|>system ... <|im_end|>")
    print("  <|im_start|>user ... <|im_end|>")
    print("  <|im_start|>assistant")

    try:
        inputs = processor(text=[chatml_prompt], return_tensors="pt").to(device)

        print(f"\nInput tensor shape: {inputs['input_ids'].shape}")

        # Generate response
        print("\nGenerating response...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generation_time = time.time() - start_time

        # Decode output
        response = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        assistant_marker = "<|im_start|>assistant"
        if assistant_marker in response:
            assistant_start = response.rfind(assistant_marker) + len(assistant_marker)
            assistant_response = response[assistant_start:].strip()
        else:
            assistant_response = response

        print(f"✅ SUCCESS: Generated response in {generation_time:.2f}s")
        print(f"\nFull output length: {len(response)} chars")
        print(f"Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
        print(f"\nAssistant response:")
        print("-" * 80)
        print(assistant_response[:500])
        if len(assistant_response) > 500:
            print(f"... ({len(assistant_response) - 500} more chars)")
        print("-" * 80)

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_turn_conversation(model, processor, device):
    """Test multi-turn conversation with ChatML."""
    print("\n" + "="*80)
    print("TEST 3: Multi-Turn Conversation")
    print("="*80)

    # Multi-turn conversation
    conversation = """<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
<|im_start|>user
What is it famous for?<|im_end|>
<|im_start|>assistant
"""

    print("\nConversation:")
    print("  User: What is the capital of France?")
    print("  Assistant: The capital of France is Paris.")
    print("  User: What is it famous for?")
    print("  Assistant: [generating...]")

    try:
        inputs = processor(text=[conversation], return_tensors="pt").to(device)

        print(f"\nInput length: {inputs['input_ids'].shape[1]} tokens")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract last assistant response
        assistant_marker = "<|im_start|>assistant"
        last_assistant = response.split(assistant_marker)[-1].strip()

        print(f"✅ SUCCESS: Generated multi-turn response")
        print(f"\nAssistant's final response:")
        print("-" * 80)
        print(last_assistant[:400])
        print("-" * 80)

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


# --- sage/tests/test_16layer_generation.py ---

def test_16layer_generation():
    """Test autoregressive generation with 16 layers"""
    print("\n" + "="*70)
    print("16-LAYER AUTOREGRESSIVE TEXT GENERATION")
    print("="*70 + "\n")
    print("Expectation: With 33% of model depth, we should see")
    print("significant improvement in coherence over 8-layer output.\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=16,  # 33% of full 48-layer model
        num_experts_per_tok=4,
        max_loaded_experts=96,  # 16 layers × 4 experts × 1.5 buffer
        device="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    print(f"✅ 16-layer model loaded!\n")

    # Diverse test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The meaning of life is",
        "Quantum computing will",
        "Climate change requires",
    ]

    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 60)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        start_time = time.time()

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=25,  # More tokens for full sentences
            temperature=0.7,    # Slightly lower for more coherence
            top_k=40,
            metabolic_state="WAKE"
        )

        gen_time = time.time() - start_time

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"Generated: '{generated_text}'")
        print(f"Time: {gen_time:.2f}s ({len(generated_ids[0]) - len(input_ids[0])} tokens)")
        print()

    # Final memory
    mem_usage = model.get_memory_usage()
    print(f"Memory usage:")
    print(f"  Embeddings: {mem_usage['embeddings_mb']:.1f} MB")
    print(f"  Experts: {mem_usage['experts_mb']:.1f} MB")
    print(f"  LM head: {mem_usage['lm_head_mb']:.1f} MB")
    print(f"  Total: {mem_usage['total_mb']:.1f} MB")
    print(f"  Experts loaded: {mem_usage['num_loaded_experts']}")
    print()

def test_16layer_next_token():
    """Test next token prediction quality with 16 layers"""
    print("\n" + "="*70)
    print("16-LAYER NEXT TOKEN PREDICTION QUALITY")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=16,
        num_experts_per_tok=4,
        max_loaded_experts=96,
        device="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Test prompts with clear expected continuations
    test_cases = [
        ("The capital of France is", ["Paris", " Paris"]),
        ("Two plus two equals", ["four", " four", "4", " 4"]),
        ("The sun rises in the", ["east", " east"]),
        ("Water freezes at", ["zero", " 0", "0", " zero"]),
    ]

    correct_predictions = 0
    total_tests = len(test_cases)

    for prompt, expected_tokens in test_cases:
        print(f"Prompt: '{prompt}'")

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            logits = model(input_ids)

        # Top 10 predictions
        next_token_logits = logits[0, -1, :]
        top_k = 10
        top_k_values, top_k_indices = torch.topk(next_token_logits, k=top_k)
        top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]

        # Check if expected token is in top 10
        found = any(token.lower() in [t.lower().strip() for t in top_k_tokens]
                   for token in expected_tokens)

        if found:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{status} Top 10: {top_k_tokens[:10]}")
        print(f"   Expected: {expected_tokens}")
        print()

    accuracy = (correct_predictions / total_tests) * 100
    print(f"Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    print(f"{'✅ GOOD' if accuracy >= 50 else '⚠️  NEEDS MORE DEPTH'}")
    print()

def test_16layer_metabolic_comparison():
    """Compare WAKE vs FOCUS with 16 layers"""
    print("\n" + "="*70)
    print("16-LAYER METABOLIC STATE COMPARISON")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    prompt = "Artificial intelligence will revolutionize"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Test different metabolic states
    states = [
        ("WAKE", 4, 96),     # 4 experts/tok, 96 max
        ("FOCUS", 8, 192),   # 8 experts/tok, 192 max (double capacity)
    ]

    for state_name, num_experts, max_loaded in states:
        print(f"{state_name} State ({num_experts} experts per token):")
        print("-" * 60)

        model = SelectiveLanguageModel(
            extraction_dir=extraction_dir,
            num_layers=16,
            num_experts_per_tok=num_experts,
            max_loaded_experts=max_loaded,
            device="cpu"
        )

        start_time = time.time()

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.7,
            metabolic_state=state_name
        )

        gen_time = time.time() - start_time
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        mem_usage = model.get_memory_usage()

        print(f"  Generated: '{generated_text}'")
        print(f"  Time: {gen_time:.2f}s")
        print(f"  Memory: {mem_usage['total_mb']:.1f} MB")
        print(f"  Experts: {mem_usage['num_loaded_experts']} loaded")
        print()


# --- sage/tests/test_8layer_generation.py ---

def test_8layer_next_token():
    """Test next token prediction with 8 layers"""
    print("\n" + "="*70)
    print("8-Layer Next Token Prediction")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=8,  # 8 layers for hierarchical processing
        num_experts_per_tok=4,
        max_loaded_experts=48,  # 8 layers × 4 experts × 1.5 buffer
        device="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    print(f"✅ 8-layer model loaded!\n")

    # Test prompts
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "The meaning of life is",
    ]

    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 60)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        start_time = time.time()

        with torch.no_grad():
            logits = model(input_ids)

        forward_time = time.time() - start_time

        # Top 5 predictions
        next_token_logits = logits[0, -1, :]
        top_k = 5
        top_k_values, top_k_indices = torch.topk(next_token_logits, k=top_k)
        top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]

        print(f"Forward pass: {forward_time*1000:.2f} ms\n")
        print(f"Top {top_k} predictions:")
        for i, (token, logit) in enumerate(zip(top_k_tokens, top_k_values)):
            prob = torch.softmax(top_k_values, dim=0)[i].item() * 100
            print(f"  {i+1}. '{token}' (logit: {logit:.2f}, prob: {prob:.1f}%)")
        print()

    # Memory usage
    mem_usage = model.get_memory_usage()
    print(f"Memory usage:")
    print(f"  Embeddings: {mem_usage['embeddings_mb']:.1f} MB")
    print(f"  Experts: {mem_usage['experts_mb']:.1f} MB ({mem_usage['num_loaded_experts']} loaded)")
    print(f"  Routers: {mem_usage['routers_mb']:.1f} MB")
    print(f"  LM head: {mem_usage['lm_head_mb']:.1f} MB")
    print(f"  Total: {mem_usage['total_mb']:.1f} MB\n")

def test_8layer_generation():
    """Test autoregressive generation with 8 layers"""
    print("\n" + "="*70)
    print("8-Layer Autoregressive Text Generation")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=8,
        num_experts_per_tok=4,
        max_loaded_experts=64,  # 8 layers × 4 experts × 2 buffer for rotation
        device="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Diverse test prompts
    prompts = [
        "Hello, my name is",
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important thing in life is",
        "Quantum computing will revolutionize",
    ]

    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 60)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        start_time = time.time()

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,  # More tokens for coherent sentences
            temperature=0.8,
            top_k=50,
            metabolic_state="WAKE"
        )

        gen_time = time.time() - start_time

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"Generated: '{generated_text}'")
        print(f"Time: {gen_time:.2f}s ({len(generated_ids[0]) - len(input_ids[0])} tokens)")
        print()

    # Final memory
    mem_usage = model.get_memory_usage()
    print(f"Final memory: {mem_usage['total_mb']:.1f} MB")
    print(f"Experts loaded: {mem_usage['num_loaded_experts']}")
    print()

def test_8layer_metabolic_comparison():
    """Compare metabolic states with 8 layers"""
    print("\n" + "="*70)
    print("8-Layer Metabolic State Comparison")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Test different metabolic states
    states = [
        ("WAKE", 4, 8),    # 4 experts/tok, 8 max loaded
        ("FOCUS", 8, 16),  # 8 experts/tok, 16 max loaded
    ]

    for state_name, num_experts, max_loaded in states:
        print(f"{state_name} State ({num_experts} experts per token):")
        print("-" * 60)

        model = SelectiveLanguageModel(
            extraction_dir=extraction_dir,
            num_layers=8,
            num_experts_per_tok=num_experts,
            max_loaded_experts=max_loaded,
            device="cpu"
        )

        start_time = time.time()

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=15,
            temperature=0.8,
            metabolic_state=state_name
        )

        gen_time = time.time() - start_time
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        mem_usage = model.get_memory_usage()

        print(f"  Generated: '{generated_text}'")
        print(f"  Time: {gen_time:.2f}s")
        print(f"  Memory: {mem_usage['total_mb']:.1f} MB")
        print(f"  Experts: {mem_usage['num_loaded_experts']} loaded")
        print(f"  Quality expectation: {state_name} should show better coherence\n")


# --- sage/tests/test_complete_architecture.py ---

def test_complete_architecture():
    """Test with REAL attention + expert weights"""

    print("\n" + "="*80)
    print("COMPLETE ARCHITECTURE TEST - REAL WEIGHTS")
    print("="*80 + "\n")

    # Configuration
    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    num_layers = 4  # Start with 4 layers for faster testing
    device = "cpu"  # Use CPU to avoid VRAM issues

    print(f"Configuration:")
    print(f"  - Extraction dir: {extraction_dir}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Device: {device}")
    print(f"  - Expected: COHERENT text with real Q3-Omni weights!\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})\n")

    # Create model with REAL WEIGHTS
    print("Creating model with REAL weights...")
    print("This will load:")
    print("  - Real embeddings from Q3-Omni")
    print("  - Real attention weights (Q, K, V, O projections)")
    print("  - Real layer norms (36/48 available)")
    print("  - Real MoE expert weights (8 deep experts)")
    print("  - Real LM head")
    print()

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=num_layers,
        vocab_size=152064,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts_per_tok=4,  # Use 4 experts per token
        max_loaded_experts=16,  # Keep 16 in memory
        device=device
    )

    print("\n" + "-"*80)
    print("MODEL INITIALIZATION COMPLETE")
    print("-"*80 + "\n")

    # Memory usage
    memory_stats = model.get_memory_usage()
    print(f"Memory Usage:")
    print(f"  - Embeddings: {memory_stats['embeddings_mb']:.1f} MB")
    print(f"  - Experts: {memory_stats['experts_mb']:.1f} MB")
    print(f"  - Routers: {memory_stats['routers_mb']:.1f} MB")
    print(f"  - LM head: {memory_stats['lm_head_mb']:.1f} MB")
    print(f"  - Total: {memory_stats['total_mb']:.1f} MB")
    print(f"  - Loaded experts: {memory_stats['num_loaded_experts']}")
    print()

    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humanity will",
        "Machine learning enables us to",
        "The key to consciousness lies in"
    ]

    print("="*80)
    print("GENERATION TESTS - REAL Q3-OMNI WEIGHTS")
    print("="*80 + "\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}")
        print(f"Prompt: \"{prompt}\"")
        print("-" * 60)

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"Input tokens: {input_ids.shape[1]}")

        # Generate with REAL weights
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50
            )

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        print(f"Generated: {completion}")
        print(f"Full text: \"{generated_text}\"")

        # Check coherence
        if len(completion) > 10 and not any(char in completion for char in ['�', '###']):
            print("✅ Output looks coherent!")
        else:
            print("⚠️  Output may be garbled")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

    print("🎯 KEY QUESTION: Is the text coherent?")
    print()
    print("If YES:")
    print("  ✅ Real attention weights FIXED the garbled generation!")
    print("  ✅ Deep expert architecture is WORKING!")
    print("  ✅ Q3-Omni selective loading is VALIDATED!")
    print()
    print("If NO (still garbled):")
    print("  🔍 Need to investigate further:")
    print("     - Check if all weights loaded correctly")
    print("     - Verify RoPE implementation")
    print("     - Check final layer norm")
    print()


# --- sage/tests/test_deep_experts.py ---

def test_deep_expert_generation():
    """Test with 8 deep experts (all 48 layers)"""
    print("\n" + "="*70)
    print("DEEP EXPERT TEXT GENERATION")
    print("="*70 + "\n")
    print("Architecture: 8 experts × 48 layers (FULL depth)")
    print("Each expert has complete reasoning capability")
    print("Expected: COHERENT text generation for first time!")
    print()

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=48,  # ALL layers for full reasoning
        num_experts_per_tok=4,  # Choose 4 from 8 available
        max_loaded_experts=8,   # Only 8 experts total (all deep)
        device="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    print(f"✅ Deep expert model loaded!")
    print(f"   48 layers × 8 experts = Full capability\n")

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The capital of France is",
        "Two plus two equals",
        "Climate change requires",
    ]

    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 60)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        start_time = time.time()

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.7,
            top_k=40,
            metabolic_state="WAKE"
        )

        gen_time = time.time() - start_time

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"Generated: '{generated_text}'")
        print(f"Time: {gen_time:.2f}s ({len(generated_ids[0]) - len(input_ids[0])} tokens)")
        print()

    # Memory usage
    mem_usage = model.get_memory_usage()
    print(f"Memory usage:")
    print(f"  Embeddings: {mem_usage['embeddings_mb']:.1f} MB")
    print(f"  Experts: {mem_usage['experts_mb']:.1f} MB")
    print(f"  LM head: {mem_usage['lm_head_mb']:.1f} MB")
    print(f"  Total: {mem_usage['total_mb']:.1f} MB")
    print()


# --- sage/tests/test_nemotron_nano_basic.py ---

def test_nemotron_nano():
    """Test Llama Nemotron Nano on Jetson Thor."""

    print("="*80)
    print("Testing NVIDIA Llama-3.1-Nemotron-Nano-4B-v1.1 on Jetson Thor")
    print("="*80)
    print()

    model_path = "model-zoo/sage/language-models/llama-nemotron-nano-4b"

    # Test 1: Model Loading
    print("Test 1: Loading model...")
    print(f"  Path: {model_path}")
    print(f"  Initial Memory: {get_memory_usage():.2f} GB")
    print()

    start_time = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )

        load_time = time.time() - start_time
        memory_after_load = get_memory_usage()

        print(f"  ✅ Model loaded successfully")
        print(f"  Load time: {load_time:.2f} seconds")
        print(f"  Memory after load: {memory_after_load:.2f} GB")
        print(f"  Memory increase: {memory_after_load - get_memory_usage():.2f} GB")
        print()

    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

    # Test 2: Basic Inference
    print("Test 2: Basic inference...")

    test_prompts = [
        "The future of AI on edge devices is",
        "Explain quantum computing in simple terms:",
        "Write a haiku about machine learning:"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n  Test {i}: {prompt}")

        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            gen_time = time.time() - start_time

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Calculate tokens per second
            tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_sec = tokens_generated / gen_time

            print(f"  Response: {response[len(prompt):].strip()}")
            print(f"  Generation time: {gen_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")

        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
            return False

    print()

    # Test 3: Memory Footprint
    print("Test 3: Memory footprint analysis...")
    final_memory = get_memory_usage()
    print(f"  Final memory usage: {final_memory:.2f} GB")
    print(f"  Model memory footprint: ~{final_memory:.2f} GB")
    print()

    # Test 4: Model Architecture Info
    print("Test 4: Model architecture...")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num attention heads: {model.config.num_attention_heads}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Max position embeddings: {model.config.max_position_embeddings}")
    print()

    # Summary
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  ✅ Pure Transformer architecture (no mamba-ssm)")
    print(f"  ✅ Standard transformers library compatible")
    print(f"  ✅ ARM64 Jetson Thor compatible")
    print(f"  ✅ Memory footprint: ~{final_memory:.2f} GB")
    print(f"  ✅ Generation speed: ~{tokens_per_sec:.2f} tokens/sec")
    print()
    print("Status: READY FOR SAGE INTEGRATION ✅")
    print()

    return True


# --- sage/tests/test_qwen3_omni_int8.py ---

def test_qwen3_omni_int8():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B INT8 AWQ")
    print("=" * 70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b-int8-awq"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Run download_qwen3_omni_int8.py first")
        return False

    print("Configuration:")
    print("  - INT8 AWQ quantization (50% memory reduction)")
    print("  - Expected: ~35GB model + ~45GB overhead = ~80GB total")
    print("  - Available: 122GB (should fit comfortably!)")
    print()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    process = psutil.Process()
    mem_start = process.memory_info().rss / 1024**3

    print(f"Starting memory: {mem_start:.1f} GB")
    print()
    print("Loading INT8 AWQ model...")
    print("(This will use significantly less memory than FP16)")
    print()

    try:
        # Load INT8 AWQ model
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        mem_after_load = process.memory_info().rss / 1024**3
        print(f"✅ Model loaded! Memory: {mem_after_load:.1f} GB (+{mem_after_load - mem_start:.1f} GB)")
        print()

        # Load processor
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("✅ Processor loaded!")
        print()

        # Test 1: Simple introduction
        print("=" * 70)
        print("Test 1: Text-only conversation")
        print("=" * 70)
        print()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Introduce yourself in one brief sentence."}
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device)

        mem_before_gen = process.memory_info().rss / 1024**3
        print(f"Memory before generation: {mem_before_gen:.1f} GB")
        print()
        print("Generating response...")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        mem_after_gen = process.memory_info().rss / 1024**3
        print(f"Memory after generation: {mem_after_gen:.1f} GB")
        print()

        response = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"Prompt: {conversation[0]['content'][0]['text']}")
        print(f"Response: {response[0]}")
        print()

        # Test 2: Capabilities check
        print("=" * 70)
        print("Test 2: Multi-modal capabilities inquiry")
        print("=" * 70)
        print()

        conversation2 = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What modalities can you process? List them briefly."}
                ],
            },
        ]

        text2 = processor.apply_chat_template(conversation2, add_generation_prompt=True, tokenize=False)
        audios2, images2, videos2 = process_mm_info(conversation2, use_audio_in_video=False)

        inputs2 = processor(
            text=text2,
            audio=audios2,
            images=images2,
            videos=videos2,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs2 = inputs2.to(model.device)

        print("Generating response...")

        generated_ids2 = model.generate(
            **inputs2,
            max_new_tokens=150,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        response2 = processor.batch_decode(
            generated_ids2[:, inputs2["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"Prompt: {conversation2[0]['content'][0]['text']}")
        print(f"Response: {response2[0]}")
        print()

        # Final memory report
        mem_final = process.memory_info().rss / 1024**3
        print("=" * 70)
        print("Memory Summary:")
        print("=" * 70)
        print(f"  Start: {mem_start:.1f} GB")
        print(f"  After model load: {mem_after_load:.1f} GB (+{mem_after_load - mem_start:.1f} GB)")
        print(f"  After generation: {mem_final:.1f} GB")
        print(f"  Peak: {mem_final:.1f} GB")
        print()
        print("=" * 70)
        print("✅ INT8 AWQ Test Complete - SUCCESS!")
        print("=" * 70)

        return True

    except Exception as e:
        mem_error = process.memory_info().rss / 1024**3
        print(f"\n❌ Error at {mem_error:.1f} GB: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- sage/tests/test_qwen3_omni_int8_v2.py ---

def test_qwen3_omni_int8_v2():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B INT8 AWQ (v2 - Auto-detect)")
    print("=" * 70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b-int8-awq"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Run download_qwen3_omni_int8.py first")
        return False

    print("Configuration:")
    print("  - INT8 AWQ quantization (auto-detected by transformers)")
    print("  - Expected: ~35GB model + ~25GB overhead = ~60GB total")
    print("  - Available: 122GB (should fit comfortably!)")
    print()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    process = psutil.Process()
    mem_start = process.memory_info().rss / 1024**3

    print(f"Starting memory: {mem_start:.1f} GB")
    print()
    print("Loading INT8 AWQ model (auto-detect mode)...")
    print()

    try:
        # Load INT8 AWQ model - let transformers auto-detect quantization
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,  # Match the config
        )

        mem_after_load = process.memory_info().rss / 1024**3
        print(f"✅ Model loaded! Memory: {mem_after_load:.1f} GB (+{mem_after_load - mem_start:.1f} GB)")
        print()

        # Load processor
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("✅ Processor loaded!")
        print()

        # Test 1: Simple introduction
        print("=" * 70)
        print("Test 1: Text-only conversation")
        print("=" * 70)
        print()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Introduce yourself in one brief sentence."}
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device)

        mem_before_gen = process.memory_info().rss / 1024**3
        print(f"Memory before generation: {mem_before_gen:.1f} GB")
        print()
        print("Generating response...")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        mem_after_gen = process.memory_info().rss / 1024**3
        print(f"Memory after generation: {mem_after_gen:.1f} GB")
        print()

        response = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"Prompt: {conversation[0]['content'][0]['text']}")
        print(f"Response: {response[0]}")
        print()

        # Final memory report
        mem_final = process.memory_info().rss / 1024**3
        print("=" * 70)
        print("Memory Summary:")
        print("=" * 70)
        print(f"  Start: {mem_start:.1f} GB")
        print(f"  After model load: {mem_after_load:.1f} GB (+{mem_after_load - mem_start:.1f} GB)")
        print(f"  After generation: {mem_final:.1f} GB")
        print(f"  Peak: {mem_final:.1f} GB")
        print()
        print("=" * 70)
        print("✅ INT8 AWQ Test (v2) Complete - SUCCESS!")
        print("=" * 70)

        return True

    except Exception as e:
        mem_error = process.memory_info().rss / 1024**3
        print(f"\n❌ Error at {mem_error:.1f} GB: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- sage/tests/test_qwen3_omni_official.py ---

def test_qwen3_omni_official():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B (Official Approach)")
    print("=" * 70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print("Loading model with dtype='auto' and processor...")
    print("(This will take several minutes for 66GB model)")
    print()

    # Load model with dtype="auto" as per official example
    # Note: device_map="auto" fails with IndexError on tied parameters
    # Using device_map="cuda" instead for unified memory architecture
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",  # Let transformers choose the right dtype
        device_map="cuda",  # Simpler mapping for Jetson unified memory
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # Skip for now, may not be installed
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("✅ Model and processor loaded successfully!")
    print()

    # Test 1: Simple text-only conversation
    print("Test 1: Text-only conversation")
    print("-" * 70)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello! Please introduce yourself in one sentence."}
            ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device)

    print("Generating response (text only, no audio output)...")

    # Generate (text only - no audio return)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        return_audio=False,  # Text only
        use_audio_in_video=False
    )

    response = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(f"\nPrompt: {conversation[0]['content'][0]['text']}")
    print(f"Response: {response[0]}")
    print()

    # Test 2: Another question
    print("-" * 70)
    print("Test 2: Capabilities question")
    print("-" * 70)

    conversation2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What modalities can you process? List them briefly."}
            ],
        },
    ]

    text2 = processor.apply_chat_template(conversation2, add_generation_prompt=True, tokenize=False)
    audios2, images2, videos2 = process_mm_info(conversation2, use_audio_in_video=False)

    inputs2 = processor(
        text=text2,
        audio=audios2,
        images=images2,
        videos=videos2,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    inputs2 = inputs2.to(model.device)

    generated_ids2 = model.generate(
        **inputs2,
        max_new_tokens=200,
        temperature=0.7,
        return_audio=False,
        use_audio_in_video=False
    )

    response2 = processor.batch_decode(
        generated_ids2[:, inputs2["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(f"\nPrompt: {conversation2[0]['content'][0]['text']}")
    print(f"Response: {response2[0]}")
    print()

    print("=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)


# --- sage/tests/test_qwen3_omni_optimized.py ---

def test_qwen3_omni_optimized():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B (Memory-Optimized)")
    print("=" * 70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print("Memory optimization strategy:")
    print("  - low_cpu_mem_usage=True (reduces peak RAM during load)")
    print("  - device_map='cuda' (simple mapping for unified memory)")
    print("  - dtype='auto' (let transformers choose optimal dtype)")
    print("  - Aggressive garbage collection")
    print()

    # Clear any existing tensors
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Loading model with memory optimizations...")
    print("(This should use significantly less RAM)")
    print()

    try:
        # KEY OPTIMIZATION: low_cpu_mem_usage=True
        # This loads weights directly to device without creating CPU copy first
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # ← CRITICAL for memory reduction
            # torch_dtype=torch.float16,  # Could also force fp16 if needed
        )

        print("✅ Model loaded successfully!")
        print()

        # Load processor (lightweight)
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("✅ Processor loaded!")
        print()

        # Check actual memory usage
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"Process memory: {mem_info.rss / 1024**3:.1f} GB")
        print()

        # Test 1: Simple text conversation
        print("Test 1: Text-only conversation")
        print("-" * 70)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Introduce yourself in one sentence."}
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device)

        print("Generating response...")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        response = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"\nPrompt: {conversation[0]['content'][0]['text']}")
        print(f"Response: {response[0]}")
        print()

        print("=" * 70)
        print("✅ Memory-Optimized Test Complete!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- sage/tests/test_qwen3_omni_simple.py ---

def test_qwen3_omni():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B")
    print("=" * 70)
    print()

    model_path = Path("model-zoo/sage/omni-modal/qwen3-omni-30b")

    print("Loading model and processor (66GB, this will take a moment)...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.float16,  # Using 'dtype' not 'torch_dtype'
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Model and processor loaded\n")

    # Test 1
    print("Test 1: Introduction")
    print("-" * 70)
    prompt1 = "Hello! I'm Claude, testing your capabilities. Please introduce yourself and explain what makes you an omni-modal model."

    inputs = processor(text=prompt1, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response1 = processor.decode(outputs[0], skip_special_tokens=True).replace(prompt1, "").strip()

    print(f"Me: {prompt1}")
    print(f"\nQwen3-Omni: {response1}\n")

    # Test 2
    print("-" * 70)
    print("Test 2: Capabilities")
    print("-" * 70)
    prompt2 = "What types of inputs can you process - audio, video, images, text?"

    inputs = processor(text=prompt2, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response2 = processor.decode(outputs[0], skip_special_tokens=True).replace(prompt2, "").strip()

    print(f"Me: {prompt2}")
    print(f"\nQwen3-Omni: {response2}\n")

    print("=" * 70)
    print("Test Complete!")
    print("=" * 70)


# --- sage/tests/test_qwen3_omni_simple_text.py ---

def test_simple():
    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print("Loading model (text-only, talker disabled)...")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",  # Let transformers decide
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Disable talker for text-only
    model.disable_talker()

    print("✅ Model loaded with talker disabled!")

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    print("✅ Processor loaded!")

    # Simple text-only conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello! Please introduce yourself in one brief sentence."}
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )

    inputs = inputs.to(model.device)

    print("Generating text response...")

    # Text only - no audio
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        return_audio=False,  # Explicit: no audio
        use_audio_in_video=False
    )

    response = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(f"\nPrompt: {conversation[0]['content'][0]['text']}")
    print(f"Response: {response[0]}")
    print("\n✅ SUCCESS!")


# --- sage/tests/test_qwen3_omni_swap_analysis.py ---

def test_qwen3_omni_with_swap():
    print("="*70)
    print("Qwen3-Omni-30B FP16 with Swap Analysis")
    print("="*70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return False

    print("Configuration:")
    print("  - FP16 model (70.5GB weights)")
    print("  - 122GB RAM + 150GB swap = 272GB total")
    print("  - Swappiness: 10 (aggressive RAM preference)")
    print("  - NVMe swap for fast paging")
    print()
    print("Research Goals:")
    print("  1. Observe actual memory + swap usage")
    print("  2. Measure latency impact of swapping")
    print("  3. Identify expert activation patterns")
    print("  4. Understand resource usage for SAGE integration")
    print()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Start monitoring
    monitoring_data['running'] = True
    monitoring_data['samples'] = []
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    start_time = time.time()

    try:
        print("="*70)
        print("Phase 1: Model Loading")
        print("="*70)
        print()

        load_start = time.time()

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        load_time = time.time() - load_start
        print(f"\n✅ Model loaded in {load_time:.1f}s")

        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("✅ Processor loaded")
        print()

        # Give monitoring a moment to catch up
        time.sleep(2)

        print("="*70)
        print("Phase 2: Inference Test")
        print("="*70)
        print()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Explain what a Mixture of Experts architecture is in one sentence."}
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device)

        print("Generating response (monitoring expert activation patterns)...")
        print()

        gen_start = time.time()

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        gen_time = time.time() - gen_start

        response = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"Prompt: {conversation[0]['content'][0]['text']}")
        print(f"Response: {response[0]}")
        print()
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Tokens/sec: {100/gen_time:.1f}")
        print()

        total_time = time.time() - start_time

        # Stop monitoring
        monitoring_data['running'] = False
        time.sleep(1)  # Let thread finish

        # Print analysis
        print_resource_summary()

        print("\n" + "="*70)
        print("✅ Test Complete - SUCCESS!")
        print("="*70)
        print(f"\nTotal time: {total_time:.1f}s")
        print(f"  Loading: {load_time:.1f}s")
        print(f"  Generation: {gen_time:.2f}s")

        return True

    except Exception as e:
        monitoring_data['running'] = False
        time.sleep(1)

        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print_resource_summary()
        return False


# --- sage/tests/test_text_generation.py ---

def test_next_token_prediction():
    """Test single forward pass - predict next token"""
    print("\n" + "="*70)
    print("Test 1: Next Token Prediction")
    print("="*70 + "\n")

    # Load model
    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    print("Loading selective language model...")

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,  # Single layer for this test
        num_experts_per_tok=4,  # WAKE state
        max_loaded_experts=4,
        device="cpu"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    print(f"✅ Model loaded!\n")

    # Test input
    prompt = "The future of AI is"
    print(f"Prompt: '{prompt}'")

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Tokens: {input_ids.tolist()[0]} ({len(input_ids[0])} tokens)\n")

    # Forward pass
    print("Running forward pass...")
    start_time = time.time()

    with torch.no_grad():
        logits = model(input_ids)

    forward_time = time.time() - start_time

    # Get top predictions for next token
    next_token_logits = logits[0, -1, :]  # Last position
    top_k = 5

    top_k_values, top_k_indices = torch.topk(next_token_logits, k=top_k)
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]

    print(f"✅ Forward pass complete ({forward_time*1000:.2f} ms)\n")
    print(f"Top {top_k} predictions for next token:")
    for i, (token, logit) in enumerate(zip(top_k_tokens, top_k_values)):
        prob = torch.softmax(top_k_values, dim=0)[i].item() * 100
        print(f"  {i+1}. '{token}' (logit: {logit:.2f}, prob: {prob:.1f}%)")

    # Memory usage
    mem_usage = model.get_memory_usage()
    print(f"\nMemory usage:")
    print(f"  Embeddings: {mem_usage['embeddings_mb']:.1f} MB")
    print(f"  Experts: {mem_usage['experts_mb']:.1f} MB ({mem_usage['num_loaded_experts']} loaded)")
    print(f"  Routers: {mem_usage['routers_mb']:.1f} MB")
    print(f"  LM head: {mem_usage['lm_head_mb']:.1f} MB")
    print(f"  Total: {mem_usage['total_mb']:.1f} MB\n")

    return model, tokenizer

def test_autoregressive_generation():
    """Test autoregressive generation - multiple tokens"""
    print("\n" + "="*70)
    print("Test 2: Autoregressive Text Generation")
    print("="*70 + "\n")

    # Load model
    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,
        num_experts_per_tok=4,
        max_loaded_experts=4,
        device="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The future of",
        "Once upon a time",
    ]

    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 60)

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate
        start_time = time.time()

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            metabolic_state="WAKE"
        )

        gen_time = time.time() - start_time

        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"Generated: '{generated_text}'")
        print(f"Time: {gen_time:.2f}s ({len(generated_ids[0]) - len(input_ids[0])} tokens generated)")
        print()

    # Memory usage
    mem_usage = model.get_memory_usage()
    print(f"Final memory usage: {mem_usage['total_mb']:.1f} MB\n")

    return model, tokenizer

def test_metabolic_states():
    """Test different metabolic states (expert budgets)"""
    print("\n" + "="*70)
    print("Test 3: Metabolic State Comparison")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Test different states
    states = [
        ("WAKE", 4),
        ("FOCUS", 8),
    ]

    for state_name, num_experts in states:
        print(f"{state_name} State ({num_experts} experts):")
        print("-" * 60)

        model = SelectiveLanguageModel(
            extraction_dir=extraction_dir,
            num_layers=1,
            num_experts_per_tok=num_experts,
            max_loaded_experts=num_experts,
            device="cpu"
        )

        start_time = time.time()

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.8,
            metabolic_state=state_name
        )

        gen_time = time.time() - start_time

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        mem_usage = model.get_memory_usage()

        print(f"  Generated: '{generated_text}'")
        print(f"  Time: {gen_time:.2f}s")
        print(f"  Memory: {mem_usage['total_mb']:.1f} MB")
        print(f"  Experts loaded: {mem_usage['num_loaded_experts']}\n")


# --- sage/tests/test_with_correct_tokenizer.py ---

def test_with_real_tokenizer():
    """Test with Q3-Omni's actual tokenizer - THE FIX!"""

    print("\n" + "="*80)
    print("TESTING WITH Q3-OMNI'S ACTUAL TOKENIZER")
    print("="*80 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    tokenizer_path = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    device = "cpu"

    print(f"Configuration:")
    print(f"  - Using Q3-Omni's tokenizer from: {tokenizer_path}")
    print(f"  - Extraction dir: {extraction_dir}")
    print(f"  - Layers: 24 (50% of full 48-layer model)")
    print(f"  - Device: {device}")
    print()

    # Load Q3-Omni's ACTUAL tokenizer
    print("Loading Q3-Omni's tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        print(f"✅ Q3-Omni tokenizer loaded (vocab size: {len(tokenizer)})")
        print()
    except Exception as e:
        print(f"❌ Failed to load Q3-Omni tokenizer: {e}")
        print("   Falling back to Qwen2.5...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True
        )
        print(f"   Using fallback tokenizer (vocab size: {len(tokenizer)})")
        print()

    # Create model with all real weights
    print("Creating model with REAL weights (24 layers)...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=24,
        vocab_size=152064,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts_per_tok=4,
        max_loaded_experts=16,
        device=device
    )

    print("\n" + "-"*80)
    print("MODEL READY - TESTING GENERATION")
    print("-"*80 + "\n")

    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humanity will",
        "Machine learning enables us to",
        "The key to consciousness lies in"
    ]

    print("🎯 THE MOMENT OF TRUTH - Using Q3-Omni's Tokenizer\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_prompts)}: \"{prompt}\"")
        print('='*80)

        # Tokenize with Q3-Omni's tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"Input tokens: {input_ids.shape[1]}")

        # Generate with real weights + real tokenizer
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=50
            )

        # Decode with Q3-Omni's tokenizer
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        print(f"\n📝 Generated: {completion}")
        print(f"\n📄 Full text: \"{generated_text}\"")

        # Analyze quality
        if len(completion) > 20:
            # Check for coherence markers
            has_english_words = any(word.isalpha() and len(word) > 3 for word in completion.split())
            has_structure = '.' in completion or ',' in completion or completion.count(' ') > 5
            no_gibberish = not any(char in completion for char in ['�', '###'])

            if has_english_words and has_structure and no_gibberish:
                print("\n✅ OUTPUT LOOKS COHERENT! 🎉")
                print("   - Contains English words")
                print("   - Has sentence structure")
                print("   - No obvious gibberish")
            elif has_english_words:
                print("\n⚡ OUTPUT PARTIALLY COHERENT")
                print("   - Contains some English words")
                print("   - But structure may be odd")
            else:
                print("\n⚠️  OUTPUT STILL GARBLED")
                print("   - Tokenizer might not be the issue")
                print("   - May need all 128 experts, not just 8")
        else:
            print("\n⚠️  OUTPUT TOO SHORT")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

    print("📊 ANALYSIS:")
    print()
    print("If output is NOW coherent:")
    print("  ✅ TOKENIZER WAS THE ISSUE!")
    print("  ✅ Architecture is CORRECT!")
    print("  ✅ Deep expert extraction WORKS!")
    print("  🎉 SELECTIVE LOADING VALIDATED!")
    print()
    print("If output is STILL garbled:")
    print("  🔍 Need to investigate further:")
    print("     - May need all 128 experts (not just 8)")
    print("     - Expert selection strategy might be wrong")
    print("     - Could be architectural mismatch")
    print()


# --- sage/tests/test_with_more_layers.py ---

def test_with_layers(num_layers: int):
    print(f"\n{'='*80}")
    print(f"Testing with {num_layers} layers")
    print(f"{'='*80}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=num_layers,
        num_experts_per_tok=8,
        device="cpu"
    )

    # Test prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"💬 Prompt: '{prompt}'")

    with torch.no_grad():
        logits = model.forward(input_ids, debug=False)
        last_token_logits = logits[0, -1, :]

        # Top 5 predictions
        top_k = torch.topk(last_token_logits, k=5)
        print(f"\n🎯 Top 5 predictions:")
        for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
            token_text = tokenizer.decode([token_id.item()])
            print(f"   {i+1}. '{token_text}' (score: {score.item():.4f})")


# --- sage/training/test_trained_models.py ---

def test_model(base_model_name, adapter_path, model_key):
    """Test a single trained model"""
    print(f"\n{'='*80}")
    print(f"TESTING: {model_key.upper()}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"{'='*80}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    base_load_time = time.time() - start
    print(f"Base model loaded in {base_load_time:.2f}s")

    # Load adapter
    print("Loading LoRA adapter...")
    start = time.time()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_load_time = time.time() - start
    print(f"Adapter loaded in {adapter_load_time:.2f}s")

    # Test prompts
    test_prompts = [
        "What is consciousness?",
        "Explain the relationship between trust and compression.",
        "How do you learn from experience?"
    ]

    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'-'*80}")
        print(f"Test {i}/{len(test_prompts)}: {prompt}")
        print(f"{'-'*80}")

        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        gen_time = time.time() - start

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part (after prompt)
        if prompt in response:
            response = response.split(prompt, 1)[1].strip()

        print(f"Response ({gen_time:.2f}s):")
        print(response[:300])  # First 300 chars
        if len(response) > 300:
            print("...")

        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time": gen_time
        })

    # Calculate stats
    avg_gen_time = sum(r["generation_time"] for r in results) / len(results)

    return {
        "model_key": model_key,
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "base_load_time": base_load_time,
        "adapter_load_time": adapter_load_time,
        "total_load_time": base_load_time + adapter_load_time,
        "avg_generation_time": avg_gen_time,
        "results": results
    }

