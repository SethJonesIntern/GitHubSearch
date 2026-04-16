# aws/sagemaker-huggingface-inference-toolkit
# 7 LLM-backed test functions across 10 test files
# Source: https://github.com/aws/sagemaker-huggingface-inference-toolkit

# --- tests/integ/test_diffusers.py ---

def test_text_to_image_model():
    image_uri = get_framework_ecr_image(repository_name="huggingface-pytorch-inference", device="gpu")

    name = "hf-test-text-to-image"
    task = "text-to-image"
    model = "echarlaix/tiny-random-stable-diffusion-xl"
    # instance_type = "ml.m5.large" if device == "cpu" else "ml.g4dn.xlarge"
    instance_type = "local_gpu"
    env = {"HF_MODEL_ID": model, "HF_TASK": task}

    sagemaker_session = Session()
    client = boto3.client("sagemaker-runtime")

    hf_model = Model(
        image_uri=image_uri,  # A Docker image URI.
        model_data=None,  # The S3 location of a SageMaker model data .tar.gz
        env=env,  # Environment variables to run with image_uri when hosted in SageMaker (default: None).
        role=SAGEMAKER_EXECUTION_ROLE,  # An AWS IAM role (either name or full ARN).
        name=name,  # The model name
        sagemaker_session=sagemaker_session,
    )

    with timeout_and_delete_by_name(name, sagemaker_session, minutes=59):
        # Use accelerator type to differentiate EI vs. CPU and GPU. Don't use processor value
        hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=name,
        )
        response = client.invoke_endpoint(
            EndpointName=name,
            Body={"inputs": "a yellow lemon tree"},
            ContentType="application/json",
            Accept="image/png",
        )

        # validate response
        response_body = response["Body"].read().decode("utf-8")

        img = Image.open(BytesIO(response_body))
        assert isinstance(img, Image.Image)

        clean_up(endpoint_name=name, sagemaker_session=sagemaker_session)


# --- tests/integ/test_models_from_hub.py ---

def test_deployment_from_hub(task, device, framework):
    image_uri = get_framework_ecr_image(repository_name=f"huggingface-{framework}-inference", device=device)
    name = f"hf-test-{framework}-{device}-{task}".replace("_", "-")
    model = task2model[task][framework]
    # instance_type = "ml.m5.large" if device == "cpu" else "ml.g4dn.xlarge"
    instance_type = "local" if device == "cpu" else "local_gpu"
    number_of_requests = 100
    if model is None:
        return

    env = {"HF_MODEL_ID": model, "HF_TASK": task}

    sagemaker_session = Session()
    client = boto3.client("sagemaker-runtime")

    hf_model = Model(
        image_uri=image_uri,  # A Docker image URI.
        model_data=None,  # The S3 location of a SageMaker model data .tar.gz
        env=env,  # Environment variables to run with image_uri when hosted in SageMaker (default: None).
        role=SAGEMAKER_EXECUTION_ROLE,  # An AWS IAM role (either name or full ARN).
        name=name,  # The model name
        sagemaker_session=sagemaker_session,
    )

    with timeout_and_delete_by_name(name, sagemaker_session, minutes=59):
        # Use accelerator type to differentiate EI vs. CPU and GPU. Don't use processor value
        hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=name,
        )

        # Keep track of the inference time
        time_buffer = []

        # Warm up the model
        if task == "image-classification":
            response = client.invoke_endpoint(
                EndpointName=name,
                Body=task2input[task],
                ContentType="image/jpeg",
                Accept="application/json",
            )
        elif task == "automatic-speech-recognition":
            response = client.invoke_endpoint(
                EndpointName=name,
                Body=task2input[task],
                ContentType="audio/x-flac",
                Accept="application/json",
            )
        else:
            response = client.invoke_endpoint(
                EndpointName=name,
                Body=json.dumps(task2input[task]),
                ContentType="application/json",
                Accept="application/json",
            )

        # validate response
        response_body = response["Body"].read().decode("utf-8")

        assert True is task2validation[task](result=json.loads(response_body), snapshot=task2output[task])

        for _ in range(number_of_requests):
            with track_infer_time(time_buffer):
                if task == "image-classification":
                    response = client.invoke_endpoint(
                        EndpointName=name,
                        Body=task2input[task],
                        ContentType="image/jpeg",
                        Accept="application/json",
                    )
                elif task == "automatic-speech-recognition":
                    response = client.invoke_endpoint(
                        EndpointName=name,
                        Body=task2input[task],
                        ContentType="audio/x-flac",
                        Accept="application/json",
                    )
                else:
                    response = client.invoke_endpoint(
                        EndpointName=name,
                        Body=json.dumps(task2input[task]),
                        ContentType="application/json",
                        Accept="application/json",
                    )
        with open(f"{name}.json", "w") as outfile:
            data = {
                "index": name,
                "framework": framework,
                "device": device,
                "model": model,
                "number_of_requests": number_of_requests,
                "average_request_time": np.mean(time_buffer),
                "max_request_time": max(time_buffer),
                "min_request_time": min(time_buffer),
                "p95_request_time": np.percentile(time_buffer, 95),
                "body": json.loads(response_body),
            }
            json.dump(data, outfile)

        assert task2performance[task][device]["average_request_time"] >= np.mean(time_buffer)

        clean_up(endpoint_name=name, sagemaker_session=sagemaker_session)


# --- tests/unit/test_decoder_encoder.py ---

def test_decode_csv_without_header():
    with pytest.raises(PredictionException):
        decoder_encoder.decode_csv(
            "where do i live?,My name is Philipp and I live in Nuremberg\r\nwhere is Berlin?,Berlin is the capital of Germany"
        )


# --- tests/unit/test_handler_service_without_context.py ---

def test_predict(inference_handler):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=MODEL,
            model_dir=tmpdirname,
        )
        inference_handler.model = get_pipeline(task=TASK, device=-1, model_dir=storage_folder)
        prediction = inference_handler.predict(INPUT, inference_handler.model)
        assert "label" in prediction[0]
        assert "score" in prediction[0]

def test_validate_and_initialize_user_module(inference_handler):
    model_dir = os.path.join(os.getcwd(), "tests/resources/model_input_predict_output_fn_without_context")
    CONTEXT = Context("", model_dir, {}, 1, -1, "1.1.4")

    inference_handler.initialize(CONTEXT)
    CONTEXT.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
    CONTEXT.metrics = MetricsStore(1, MODEL)

    prediction = inference_handler.handle([{"body": b""}], CONTEXT)
    assert "output" in prediction[0]

    assert inference_handler.load({}) == "Loading inference_tranform_fn.py"


# --- tests/unit/test_handler_service_with_context.py ---

def test_handle(inference_handler):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=MODEL,
            model_dir=tmpdirname,
        )
        CONTEXT = Context(MODEL, storage_folder, {}, 1, -1, "1.1.4")
        CONTEXT.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
        CONTEXT.metrics = MetricsStore(1, MODEL)

        inference_handler.initialize(CONTEXT)
        json_data = json.dumps(INPUT)
        prediction = inference_handler.handle([{"body": json_data.encode()}], CONTEXT)
        loaded_response = json.loads(prediction[0])
        assert "entity" in loaded_response[0]
        assert "score" in loaded_response[0]

def test_validate_and_initialize_user_module(inference_handler):
    model_dir = os.path.join(os.getcwd(), "tests/resources/model_input_predict_output_fn_with_context")
    CONTEXT = Context("", model_dir, {}, 1, -1, "1.1.4")

    inference_handler.initialize(CONTEXT)
    CONTEXT.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
    CONTEXT.metrics = MetricsStore(1, MODEL)

    prediction = inference_handler.handle([{"body": b""}], CONTEXT)
    assert "output" in prediction[0]

    assert inference_handler.load({}, CONTEXT) == "model"
    assert inference_handler.preprocess({}, "", CONTEXT) == "data"
    assert inference_handler.predict({}, "model", CONTEXT) == "output"
    assert inference_handler.postprocess("output", "", CONTEXT) == "output"

