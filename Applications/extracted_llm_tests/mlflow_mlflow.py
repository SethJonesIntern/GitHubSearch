# mlflow/mlflow
# 335 LLM-backed test functions across 752 test files
# Source: https://github.com/mlflow/mlflow

# --- examples/sktime/test_sktime_model_export.py ---

def test_auto_arima_model_save_and_load(auto_arima_model, model_path, serialization_format):
    flavor.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = flavor.load_model(
        model_uri=model_path,
    )

    np.testing.assert_array_equal(auto_arima_model.predict(fh=FH), loaded_model.predict(fh=FH))

def test_auto_arima_model_pyfunc_output(auto_arima_model, model_path, serialization_format):
    flavor.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path)

    model_predict = auto_arima_model.predict(fh=FH)
    predict_conf = pd.DataFrame([{"fh": FH, "predict_method": "predict"}])
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)
    np.testing.assert_array_equal(model_predict, pyfunc_predict)

    model_predict_interval = auto_arima_model.predict_interval(fh=FH, coverage=COVERAGE)
    predict_interval_conf = pd.DataFrame([
        {
            "fh": FH,
            "predict_method": "predict_interval",
            "coverage": COVERAGE,
        }
    ])
    pyfunc_predict_interval = loaded_pyfunc.predict(predict_interval_conf)
    np.testing.assert_array_equal(model_predict_interval.values, pyfunc_predict_interval.values)

    model_predict_quantiles = auto_arima_model.predict_quantiles(fh=FH, alpha=ALPHA)
    predict_quantiles_conf = pd.DataFrame([
        {
            "fh": FH,
            "predict_method": "predict_quantiles",
            "alpha": ALPHA,
        }
    ])
    pyfunc_predict_quantiles = loaded_pyfunc.predict(predict_quantiles_conf)
    np.testing.assert_array_equal(model_predict_quantiles.values, pyfunc_predict_quantiles.values)

    model_predict_var = auto_arima_model.predict_var(fh=FH, cov=COV)
    predict_var_conf = pd.DataFrame([{"fh": FH, "predict_method": "predict_var", "cov": COV}])
    pyfunc_predict_var = loaded_pyfunc.predict(predict_var_conf)
    np.testing.assert_array_equal(model_predict_var.values, pyfunc_predict_var.values)

def test_naive_forecaster_model_with_regressor_pyfunc_output(
    naive_forecaster_model_with_regressor, model_path, data_longley
):
    _, _, _, X_test = data_longley

    flavor.save_model(sktime_model=naive_forecaster_model_with_regressor, path=model_path)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path)

    X_test_array = convert(X_test, "pd.DataFrame", "np.ndarray")

    model_predict = naive_forecaster_model_with_regressor.predict(fh=FH, X=X_test)
    predict_conf = pd.DataFrame([{"fh": FH, "predict_method": "predict", "X": X_test_array}])
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)
    np.testing.assert_array_equal(model_predict, pyfunc_predict)

    model_predict_interval = naive_forecaster_model_with_regressor.predict_interval(
        fh=FH, coverage=COVERAGE, X=X_test
    )
    predict_interval_conf = pd.DataFrame([
        {
            "fh": FH,
            "predict_method": "predict_interval",
            "coverage": COVERAGE,
            "X": X_test_array,
        }
    ])
    pyfunc_predict_interval = loaded_pyfunc.predict(predict_interval_conf)
    np.testing.assert_array_equal(model_predict_interval.values, pyfunc_predict_interval.values)

    model_predict_quantiles = naive_forecaster_model_with_regressor.predict_quantiles(
        fh=FH, alpha=ALPHA, X=X_test
    )
    predict_quantiles_conf = pd.DataFrame([
        {
            "fh": FH,
            "predict_method": "predict_quantiles",
            "alpha": ALPHA,
            "X": X_test_array,
        }
    ])
    pyfunc_predict_quantiles = loaded_pyfunc.predict(predict_quantiles_conf)
    np.testing.assert_array_equal(model_predict_quantiles.values, pyfunc_predict_quantiles.values)

    model_predict_var = naive_forecaster_model_with_regressor.predict_var(fh=FH, cov=COV, X=X_test)
    predict_var_conf = pd.DataFrame([
        {
            "fh": FH,
            "predict_method": "predict_var",
            "cov": COV,
            "X": X_test_array,
        }
    ])
    pyfunc_predict_var = loaded_pyfunc.predict(predict_var_conf)
    np.testing.assert_array_equal(model_predict_var.values, pyfunc_predict_var.values)

def test_signature_and_examples_saved_correctly(
    auto_arima_model, data_airline, model_path, use_signature, use_example
):
    # Note: Signature inference fails on native model predict_interval/predict_quantiles
    prediction = auto_arima_model.predict(fh=FH)
    signature = infer_signature(data_airline, prediction) if use_signature else None
    example = pd.DataFrame(data_airline[0:5].copy(deep=False)) if use_example else None
    flavor.save_model(auto_arima_model, path=model_path, signature=signature, input_example=example)
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)

def test_predict_var_signature_saved_correctly(
    auto_arima_model, data_airline, model_path, use_signature
):
    prediction = auto_arima_model.predict_var(fh=FH)
    signature = infer_signature(data_airline, prediction) if use_signature else None
    flavor.save_model(auto_arima_model, path=model_path, signature=signature)
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature

def test_signature_and_example_for_pyfunc_predict_interval(
    auto_arima_model, model_path, data_airline, use_signature, use_example
):
    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")
    flavor.save_model(sktime_model=auto_arima_model, path=model_path_primary)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path_primary)
    predict_conf = pd.DataFrame([
        {
            "fh": FH,
            "predict_method": "predict_interval",
            "coverage": COVERAGE,
        }
    ])
    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(data_airline, forecast) if use_signature else None
    example = pd.DataFrame(data_airline[0:5].copy(deep=False)) if use_example else None
    flavor.save_model(
        auto_arima_model,
        path=model_path_secondary,
        signature=signature,
        input_example=example,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path_secondary).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)

def test_signature_for_pyfunc_predict_quantiles(
    auto_arima_model, model_path, data_airline, use_signature
):
    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")
    flavor.save_model(sktime_model=auto_arima_model, path=model_path_primary)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path_primary)
    predict_conf = pd.DataFrame([
        {
            "fh": FH,
            "predict_method": "predict_quantiles",
            "alpha": ALPHA,
        }
    ])
    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(data_airline, forecast) if use_signature else None
    flavor.save_model(auto_arima_model, path=model_path_secondary, signature=signature)
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature

def test_log_model(auto_arima_model, tmp_path, should_start_run, serialization_format):
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "sktime"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
        model_info = flavor.log_model(
            sktime_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = flavor.load_model(
            model_uri=model_uri,
        )
        np.testing.assert_array_equal(auto_arima_model.predict(), reloaded_model.predict())
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
    finally:
        mlflow.end_run()

def test_sktime_pyfunc_raises_invalid_df_input(auto_arima_model, model_path):
    flavor.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(MlflowException, match="The provided prediction pd.DataFrame "):
        loaded_pyfunc.predict(pd.DataFrame([{"predict_method": "predict"}, {"fh": FH}]))

    with pytest.raises(MlflowException, match="The provided prediction configuration "):
        loaded_pyfunc.predict(pd.DataFrame([{"invalid": True}]))

    with pytest.raises(MlflowException, match="Invalid `predict_method` value"):
        loaded_pyfunc.predict(pd.DataFrame([{"predict_method": "predict_proba"}]))


# --- tests/agno/test_agno_tracing.py ---

def test_function_execute_failure_tracing():
    from agno.exceptions import AgentRunException

    def boom(x):
        raise AgentRunException("bad")

    fc = FunctionCall(function=Function.from_callable(boom, name="boom"), arguments={"x": 1})

    mlflow.agno.autolog(log_traces=True)
    with pytest.raises(AgentRunException, match="bad"):
        fc.execute()

    trace = get_traces()[0]
    assert trace.info.status == "ERROR"
    span = trace.data.spans[0]
    assert span.span_type == SpanType.TOOL
    assert span.status.status_code == SpanStatusCode.ERROR
    assert span.inputs == {"x": 1}
    assert span.outputs is None


# --- tests/bedrock/test_bedrock_autolog.py ---

def test_bedrock_autolog_invoke_model_capture_exception():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    request_body = json.dumps({
        # Invalid user role to trigger an exception
        "messages": [{"role": "invalid-user", "content": "Hi"}],
        "max_tokens": 300,
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 0.1,
        "top_p": 0.9,
    })

    with pytest.raises(NoCredentialsError, match="Unable to locate credentials"):
        client.invoke_model(
            body=request_body,
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.invoke_model"
    assert span.status.status_code == "ERROR"
    assert span.inputs == {
        "body": request_body,
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    }
    assert span.outputs is None
    assert span.model_name == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert len(span.events) == 1
    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.message"].startswith("Unable to locate credentials")

def test_bedrock_autolog_converse_error():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    with pytest.raises(NoCredentialsError, match="Unable to locate credentials"):
        client.converse(**_CONVERSE_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"

    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.converse"
    assert span.status.status_code == "ERROR"
    assert span.inputs == _CONVERSE_REQUEST
    assert span.outputs is None
    assert span.model_name == _CONVERSE_REQUEST["modelId"]
    assert len(span.events) == 1


# --- tests/dspy/test_dspy_autolog.py ---

def test_autolog_tracing_during_compilation_disabled_by_default():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM({
            "What is 1 + 1?": {"answer": "2"},
            "What is 2 + 2?": {"answer": "1000"},
        })
    )

    # Samples from HotpotQA dataset
    trainset = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]

    program = Predict("question -> answer")

    # Compile should NOT generate traces by default
    teleprompter = BootstrapFewShot()
    teleprompter.compile(program, trainset=trainset)

    assert len(get_traces()) == 0

    # If opted in, traces should be generated during compilation
    mlflow.dspy.autolog(log_traces_from_compile=True)

    teleprompter.compile(program, trainset=trainset)

    traces = get_traces()
    assert len(traces) == 2
    assert all(trace.info.status == "OK" for trace in traces)

    # Opt-out again
    mlflow.dspy.autolog(log_traces_from_compile=False)

    teleprompter.compile(program, trainset=trainset)
    assert len(get_traces()) == 2  # no new traces

def test_autolog_tracing_during_evaluation_enabled_by_default():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM({
            "What is 1 + 1?": {"answer": "2"},
            "What is 2 + 2?": {"answer": "1000"},
        })
    )

    # Samples from HotpotQA dataset
    trainset = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]

    program = Predict("question -> answer")

    # Evaluate should generate traces by default
    evaluator = Evaluate(devset=trainset)
    eval_res = evaluator(program, metric=answer_exact_match)

    score = eval_res if isinstance(eval_res, float) else eval_res.score
    assert score == 50.0
    traces = get_traces()
    assert len(traces) == 2
    assert all(trace.info.status == "OK" for trace in traces)

    # If opted out, traces should NOT be generated during evaluation
    mlflow.dspy.autolog(log_traces_from_eval=False)

    score = evaluator(program, metric=answer_exact_match)
    assert len(get_traces()) == 2  # no new traces

def test_autolog_set_retriever_schema():
    from mlflow.models.dependencies_schemas import DependenciesSchemasType, _clear_retriever_schema

    mlflow.dspy.autolog()
    dspy.settings.configure(
        lm=DummyLM([{"answer": answer, "reasoning": "reason"} for answer in ["4", "6", "8", "10"]])
    )

    with mlflow.start_run():
        model_info = mlflow.dspy.log_model(CoT(), name="model")

    # Reset retriever schema
    _clear_retriever_schema()

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_model.predict({"question": "What is 2 + 2?"})

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert json.loads(trace.info.tags[DependenciesSchemasType.RETRIEVERS.value]) == [
        {
            "name": "retriever",
            "primary_key": "id",
            "text_column": "text",
            "doc_uri": "source",
            "other_columns": [],
        }
    ]

def test_dspy_auto_tracing_in_databricks_model_serving(with_dependencies_schema):
    from mlflow.models.dependencies_schemas import DependenciesSchemasType

    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "answer": "test output",
                    "reasoning": "No more responses",
                },
            ]
            * 2
        )
    )

    if with_dependencies_schema:
        mlflow.models.set_retriever_schema(
            primary_key="primary-key",
            text_column="text-column",
            doc_uri="doc-uri",
            other_columns=["column1", "column2"],
        )

    input_example = "What castle did David Gregory inherit?"

    with mlflow.start_run():
        model_info = mlflow.dspy.log_model(RAG(), name="model", input_example=input_example)

    databricks_request_id, response, trace_dict = score_in_model_serving(
        model_info.model_uri,
        input_example,
    )

    trace = Trace.from_dict(trace_dict)
    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.client_request_id == databricks_request_id
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 8
    assert [span.name for span in spans] == [
        "RAG.forward",
        "retrieve_context",
        "DummyRetriever.forward",
        "ChainOfThought.forward",
        "Predict.forward",
        "ChatAdapter.format",
        "DummyLM.__call__",
        "ChatAdapter.parse",
    ]

    if with_dependencies_schema:
        assert json.loads(trace.info.tags[DependenciesSchemasType.RETRIEVERS.value]) == [
            {
                "name": "retriever",
                "primary_key": "primary-key",
                "text_column": "text-column",
                "doc_uri": "doc-uri",
                "other_columns": ["column1", "column2"],
            }
        ]

def test_autolog_log_compile(log_compiles):
    class DummyOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program, kwarg1=None, kwarg2=None):
            callback = dspy.settings.callbacks[0]
            assert callback.optimizer_stack_level == 1
            return program

    mlflow.dspy.autolog(log_compiles=log_compiles)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    program = dspy.ChainOfThought("question -> answer")
    optimizer = DummyOptimizer()

    optimizer.compile(program, kwarg1=1, kwarg2="2")

    assert dspy.settings.callbacks[0].optimizer_stack_level == 0
    if log_compiles:
        run = mlflow.last_active_run()
        assert run is not None
        assert run.data.params == {
            "kwarg1": "1",
            "kwarg2": "2",
            "lm_params": json.dumps({
                "cache": True,
                "max_tokens": 1000,
                "model": "dummy",
                "model_type": "chat",
                "temperature": 0.0,
            }),
        }
        client = MlflowClient()
        artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
        assert "best_model.json" in artifacts

        # verify that a dummy model output is logged
        run = client.get_run(run.info.run_id)
        assert len(run.outputs.model_outputs) == 1
        assert isinstance(run.outputs.model_outputs[0], LoggedModelOutput)
    else:
        assert mlflow.last_active_run() is None

def test_autolog_log_compile_log_model_output_when_failure(log_compiles):
    class DummyOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program, kwarg1=None, kwarg2=None):
            raise Exception("test error")

    mlflow.dspy.autolog(log_compiles=log_compiles)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    program = dspy.ChainOfThought("question -> answer")
    optimizer = DummyOptimizer()

    with pytest.raises(Exception, match="test error"):
        optimizer.compile(program, kwarg1=1, kwarg2="2")

    if log_compiles:
        run = mlflow.last_active_run()
        assert run is not None

        # verify that a dummy model output is logged even when compilation fails
        client = MlflowClient()
        run = client.get_run(run.info.run_id)
        assert len(run.outputs.model_outputs) == 1
        assert isinstance(run.outputs.model_outputs[0], LoggedModelOutput)
    else:
        assert mlflow.last_active_run() is None

def test_autolog_log_compile_disable():
    class DummyOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program):
            return program

    mlflow.dspy.autolog(log_compiles=True)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    program = dspy.ChainOfThought("question -> answer")
    optimizer = DummyOptimizer()

    optimizer.compile(program)

    run = mlflow.last_active_run()
    assert run is not None

    # verify that run is not created when disabling autologging
    mlflow.dspy.autolog(disable=True)
    optimizer.compile(program)
    client = MlflowClient()
    runs = client.search_runs(run.info.experiment_id)
    assert len(runs) == 1

def test_autolog_log_nested_compile():
    class NestedOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program):
            callback = dspy.settings.callbacks[0]
            assert callback.optimizer_stack_level == 2
            return program

    class DummyOptimizer(dspy.teleprompt.Teleprompter):
        def __init__(self):
            super().__init__()
            self.nested_optimizer = NestedOptimizer()

        def compile(self, program):
            self.nested_optimizer.compile(program)
            callback = dspy.settings.callbacks[0]
            assert callback.optimizer_stack_level == 1
            return program

    mlflow.dspy.autolog(log_compiles=True)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    program = dspy.ChainOfThought("question -> answer")
    optimizer = DummyOptimizer()

    optimizer.compile(program)

    assert dspy.settings.callbacks[0].optimizer_stack_level == 0
    run = mlflow.last_active_run()
    assert run is not None
    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
    assert "best_model.json" in artifacts

def test_autolog_log_evals(
    tmp_path, log_evals, return_outputs, lm, examples, expected_result_table
):
    mlflow.dspy.autolog(log_evals=log_evals)

    with dspy.context(lm=lm):
        program = Predict("question -> answer")
        if is_2_7_or_newer:
            evaluator = Evaluate(devset=examples, metric=answer_exact_match)
        else:
            # return_outputs arg does not exist after 2.7
            evaluator = Evaluate(
                devset=examples, metric=answer_exact_match, return_outputs=return_outputs
            )
        evaluator(program, devset=examples)

    run = mlflow.last_active_run()
    if log_evals:
        assert run is not None
        assert run.data.metrics == {"eval": 50.0}
        assert run.data.params == {
            "Predict.signature.fields.0.description": "${question}",
            "Predict.signature.fields.0.prefix": "Question:",
            "Predict.signature.fields.1.description": "${answer}",
            "Predict.signature.fields.1.prefix": "Answer:",
            "Predict.signature.instructions": "Given the fields `question`, produce the fields `answer`.",  # noqa: E501
            "lm_params": json.dumps({
                "cache": True,
                "max_tokens": 1000,
                "model": "dummy",
                "model_type": "chat",
                "temperature": 0.0,
            }),
        }
        client = MlflowClient()
        artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
        assert "model.json" in artifacts
        if is_2_7_or_newer:
            assert "result_table.json" in artifacts
            client.download_artifacts(
                run_id=run.info.run_id, path="result_table.json", dst_path=tmp_path
            )
            result_table = json.loads((tmp_path / "result_table.json").read_text())
            assert result_table == expected_result_table
    else:
        assert run is None

def test_autolog_log_evals_disable_by_caller():
    mlflow.dspy.autolog(log_evals=True)
    examples = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
    ]
    evaluator = Evaluate(devset=examples, metric=answer_exact_match)
    program = Predict("question -> answer")
    with dspy.context(lm=DummyLM([{"answer": "2"}])):
        evaluator(program, devset=examples, callback_metadata={"disable_logging": True})

    assert mlflow.last_active_run() is None

def test_autolog_nested_evals():
    lm = DummyLM({
        "What is 1 + 1?": {"answer": "2"},
        "What is 2 + 2?": {"answer": "4"},
    })
    dspy.settings.configure(lm=lm)
    examples = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="2").with_inputs("question"),
    ]
    program = Predict("question -> answer")
    evaluator = Evaluate(devset=examples, metric=answer_exact_match)

    mlflow.dspy.autolog(log_evals=True)
    with mlflow.start_run() as active_run:
        evaluator(program, devset=examples[:1])
        evaluator(program, devset=examples[1:])

    client = MlflowClient()
    run = client.get_run(active_run.info.run_id)
    assert run.data.metrics == {"eval": 0.0}

    artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
    assert "model.json" in artifacts

    metric_history = client.get_metric_history(run.info.run_id, "eval")
    assert [metric.value for metric in metric_history] == [100.0, 0.0]

    child_runs = client.search_runs(
        run.info.experiment_id,
        filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'",
        order_by=["attributes.start_time ASC"],
    )

    assert len(child_runs) == 0

def test_autolog_log_traces_from_evals(call_args):
    mlflow.dspy.autolog(log_evals=True, log_traces_from_eval=True)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    class DummyProgram(dspy.Module):
        def forward(self, question):
            return dspy.Prediction(answer="2")

    examples = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]

    program = DummyProgram()
    evaluator = Evaluate(devset=examples, metric=answer_exact_match)

    if call_args == "args":
        result = evaluator(program, answer_exact_match, examples)
    elif call_args == "kwargs":
        result = evaluator(program=program, devset=examples, metric=answer_exact_match)
    else:
        result = evaluator(program, answer_exact_match, devset=examples)

    if _DSPY_VERSION >= Version("3.0.0"):
        from dspy.evaluate.evaluate import EvaluationResult

        assert isinstance(result, EvaluationResult)
    else:
        assert result is not None

    traces = get_traces()
    assert len(traces) == 2
    assert all(trace.info.status == "OK" for trace in traces)

    actual_values = []

    assessments = traces[0].info.assessments
    assert len(assessments) == 1
    assert isinstance(assessments[0], Feedback)
    assert assessments[0].name == "answer_exact_match"
    actual_values.append(assessments[0].value)

    assessments = traces[1].info.assessments
    assert len(assessments) == 1
    assert isinstance(assessments[0], Feedback)
    assert assessments[0].name == "answer_exact_match"
    actual_values.append(assessments[0].value)

    assert set(actual_values) == {True, False}

def test_autolog_log_traces_from_evals_log_error_assessment():
    mlflow.dspy.autolog(log_evals=True, log_traces_from_eval=True)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    class DummyProgram(dspy.Module):
        def forward(self, question):
            return dspy.Prediction(answer="2")

    def error_metric(program, devset):
        raise Exception("Error")

    examples = [Example(question="What is 1 + 1?", answer="2").with_inputs("question")]

    program = DummyProgram()
    evaluator = Evaluate(devset=examples, metric=error_metric)
    evaluator(program, error_metric, examples)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    assessments = traces[0].info.assessments
    assert len(assessments) == 1
    assert isinstance(assessments[0], Feedback)
    assert assessments[0].name == "error_metric"
    assert assessments[0].value is None
    assert assessments[0].error.error_code == "Exception"
    assert assessments[0].error.error_message == "Error"
    assert assessments[0].error.stack_trace is not None

def test_autolog_log_compile_with_evals():
    class EvalOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program, eval, trainset, valset):
            eval(program, devset=valset, callback_metadata={"metric_key": "eval_full"})
            eval(program, devset=trainset[:1], callback_metadata={"metric_key": "eval_minibatch"})
            eval(program, devset=valset, callback_metadata={"metric_key": "eval_full"})
            eval(program, devset=trainset[:1], callback_metadata={"metric_key": "eval_minibatch"})
            return program

    dspy.settings.configure(
        lm=DummyLM({
            "What is 1 + 1?": {"answer": "2"},
            "What is 2 + 2?": {"answer": "1000"},
        })
    )
    dataset = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]
    program = Predict("question -> answer")
    evaluator = Evaluate(devset=dataset, metric=answer_exact_match)
    optimizer = EvalOptimizer()

    mlflow.dspy.autolog(log_compiles=True, log_evals=True)
    optimizer.compile(program, evaluator, trainset=dataset, valset=dataset)

    # callback state
    callback = dspy.settings.callbacks[0]
    assert callback.optimizer_stack_level == 0
    assert callback._call_id_to_metric_key == {}
    assert callback._evaluation_counter == {}

    # root run
    root_run = mlflow.last_active_run()
    assert root_run is not None
    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(root_run.info.run_id))
    assert "best_model.json" in artifacts
    assert "trainset.json" in artifacts
    assert "valset.json" in artifacts
    assert root_run.data.metrics == {
        "eval_full": 50.0,
        "eval_minibatch": 100.0,
    }

    # children runs
    child_runs = client.search_runs(
        root_run.info.experiment_id,
        filter_string=f"tags.mlflow.parentRunId = '{root_run.info.run_id}'",
        order_by=["attributes.start_time ASC"],
    )
    assert len(child_runs) == 4

    for i, run in enumerate(child_runs):
        if i % 2 == 0:
            assert run.data.metrics == {"eval": 50.0}
        else:
            assert run.data.metrics == {"eval": 100.0}
        artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
        assert "model.json" in artifacts
        assert run.data.params == {
            "Predict.signature.fields.0.description": "${question}",
            "Predict.signature.fields.0.prefix": "Question:",
            "Predict.signature.fields.1.description": "${answer}",
            "Predict.signature.fields.1.prefix": "Answer:",
            "Predict.signature.instructions": "Given the fields `question`, produce the fields `answer`.",  # noqa: E501
            "lm_params": json.dumps({
                "cache": True,
                "max_tokens": 1000,
                "model": "dummy",
                "model_type": "chat",
                "temperature": 0.0,
            }),
        }

def test_autolog_link_traces_loaded_model_custom_module():
    mlflow.dspy.autolog()
    dspy.settings.configure(
        lm=DummyLM([{"answer": "test output", "reasoning": "No more responses"}] * 5)
    )
    dspy_model = CoT()

    model_infos = []
    for _ in range(5):
        with mlflow.start_run():
            model_infos.append(mlflow.dspy.log_model(dspy_model, name="model", pip_requirements=[]))

    for model_info in model_infos:
        loaded_model = mlflow.dspy.load_model(model_info.model_uri)
        loaded_model(model_info.model_id)

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        model_id = json.loads(trace.data.request)["args"][0]
        assert model_id == trace.info.request_metadata[TraceMetadataKey.MODEL_ID]

def test_autolog_link_traces_loaded_model_custom_module_pyfunc():
    mlflow.dspy.autolog()
    dspy.settings.configure(
        lm=DummyLM([{"answer": "test output", "reasoning": "No more responses"}] * 5)
    )
    dspy_model = CoT()

    model_infos = []
    for _ in range(5):
        with mlflow.start_run():
            model_infos.append(mlflow.dspy.log_model(dspy_model, name="model", pip_requirements=[]))

    for model_info in model_infos:
        pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
        pyfunc_model.predict(model_info.model_id)

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        model_id = json.loads(trace.data.request)["args"][0]
        assert model_id == trace.info.request_metadata[TraceMetadataKey.MODEL_ID]

def test_autolog_link_traces_active_model():
    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(model_id=model.model_id)
    mlflow.dspy.autolog()
    dspy.settings.configure(
        lm=DummyLM([{"answer": "test output", "reasoning": "No more responses"}] * 5)
    )
    dspy_model = CoT()

    model_infos = []
    for _ in range(5):
        with mlflow.start_run():
            model_infos.append(mlflow.dspy.log_model(dspy_model, name="model", pip_requirements=[]))

    for model_info in model_infos:
        pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
        pyfunc_model.predict(model_info.model_id)

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        model_id = json.loads(trace.data.request)["args"][0]
        assert model_id != model.model_id
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == model.model_id

def test_autolog_databricks_rm_retriever():
    mlflow.dspy.autolog()

    dspy.settings.configure(lm=DummyLM([{"output": "test output"}]))

    class DatabricksRM(dspy.Retrieve):
        def __init__(self, retrieve_uri):
            self.retrieve_uri = retrieve_uri

        def forward(self, query) -> list[str]:
            time.sleep(0.1)
            return dspy.Prediction(
                docs=["doc1", "doc2"],
                doc_ids=["id1", "id2"],
                doc_uris=["uri1", "uri2"] if self.retrieve_uri else None,
                extra_columns=[{"author": "Jim"}, {"author": "tom"}],
            )

    DatabricksRM.__module__ = "dspy.retrieve.databricks_rm"

    for retrieve_uri in [False, True]:
        retriever = DatabricksRM(retrieve_uri)
        result = retriever(query="test query")
        assert isinstance(result, dspy.Prediction)

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        assert trace is not None
        assert trace.info.status == "OK"
        assert trace.info.execution_time_ms > 0

        spans = trace.data.spans
        assert len(spans) == 1
        assert spans[0].name == "DatabricksRM.forward"
        assert spans[0].span_type == SpanType.RETRIEVER
        assert spans[0].status.status_code == "OK"
        assert spans[0].inputs == {"query": "test query"}

        if retrieve_uri:
            uri1 = "uri1"
            uri2 = "uri2"
        else:
            uri1 = None
            uri2 = None

        assert spans[0].outputs == [
            {
                "page_content": "doc1",
                "metadata": {"doc_id": "id1", "doc_uri": uri1, "author": "Jim"},
                "id": "id1",
            },
            {
                "page_content": "doc2",
                "metadata": {"doc_id": "id2", "doc_uri": uri2, "author": "tom"},
                "id": "id2",
            },
        ]


# --- tests/gateway/test_openai_compatibility.py ---

def test_chat_invalid_endpoint(client):
    with pytest.raises(openai.BadRequestError, match="is not a chat endpoint"):
        client.chat.completions.create(
            model="completions", messages=[{"role": "user", "content": "hello"}]
        )

def test_completions_invalid_endpoint(client):
    with pytest.raises(openai.BadRequestError, match="is not a completions endpoint"):
        client.completions.create(model="chat", prompt="hello")


# --- tests/genai/judges/adapters/test_litellm_adapter.py ---

def test_litellm_adapter_rejects_base_url_for_databricks(model_provider):
    from mlflow.genai.judges.adapters.litellm_adapter import LiteLLMAdapter

    adapter = LiteLLMAdapter()
    input_params = AdapterInvocationInput(
        prompt="test prompt",
        assessment_name="test",
        model_uri=f"{model_provider}:/test-endpoint",
        trace=None,
        num_retries=3,
        base_url="http://proxy:8080",
    )

    with pytest.raises(MlflowException, match="base_url and extra_headers are not supported"):
        adapter.invoke(input_params)

def test_litellm_adapter_rejects_extra_headers_for_databricks(model_provider):
    from mlflow.genai.judges.adapters.litellm_adapter import LiteLLMAdapter

    adapter = LiteLLMAdapter()
    input_params = AdapterInvocationInput(
        prompt="test prompt",
        assessment_name="test",
        model_uri=f"{model_provider}:/test-endpoint",
        trace=None,
        num_retries=3,
        extra_headers={"X-Key": "val"},
    )

    with pytest.raises(MlflowException, match="base_url and extra_headers are not supported"):
        adapter.invoke(input_params)


# --- tests/genai/judges/optimizers/test_dspy_base.py ---

def test_concrete_implementation_required():
    class IncompleteDSPyOptimizer(DSPyAlignmentOptimizer):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteDSPyOptimizer()

def test_create_judge_from_dspy_program_preserves_feedback_value_type():
    optimizer = ConcreteDSPyOptimizer()
    judge = make_judge(
        name="test_judge",
        instructions="Check {{inputs}} vs {{outputs}}",
        model="openai:/gpt-4",
        feedback_value_type=bool,
    )

    program = dspy.Predict("inputs, outputs -> result, rationale")
    program.signature.instructions = "Check {{inputs}} vs {{outputs}}"

    result = optimizer._create_judge_from_dspy_program(program, judge)

    assert result.feedback_value_type == bool


# --- tests/genai/utils/test_trace_utils.py ---

def test_convert_predict_fn(predict_fn_generator, with_tracing, should_be_wrapped):
    predict_fn = predict_fn_generator(with_tracing=with_tracing)
    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    # predict_fn is callable as is
    result = predict_fn(**sample_input)
    assert result == "test"
    assert len(get_traces()) == (1 if with_tracing else 0)
    purge_traces()

    converted_fn = convert_predict_fn(predict_fn, sample_input)

    # converted function takes a single 'request' argument
    result = converted_fn(request=sample_input)
    assert result == "test"

    # Trace should be generated if decorated or wrapped with @mlflow.trace
    assert len(get_traces()) == (1 if with_tracing or should_be_wrapped else 0)
    purge_traces()

    # All function should generate a trace when executed through mlflow.genai.evaluate
    @scorer
    def dummy_scorer(inputs, outputs):
        return 0

    mlflow.genai.evaluate(
        data=[{"inputs": sample_input}],
        predict_fn=predict_fn,
        scorers=[dummy_scorer],
    )
    assert len(get_traces()) == 1

def test_extract_expectations_from_trace_with_source_filter():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is an open source platform"})

    trace_id = span.trace_id

    human_expectation = Expectation(
        name="human_expectation",
        value={"expected": "Answer from human"},
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=human_expectation)

    llm_expectation = Expectation(
        name="llm_expectation",
        value="LLM generated expectation",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=llm_expectation)

    code_expectation = Expectation(
        name="code_expectation",
        value=42,
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=code_expectation)

    trace = mlflow.get_trace(trace_id)

    result = extract_expectations_from_trace(trace, source_type=None)
    assert result == {
        "human_expectation": {"expected": "Answer from human"},
        "llm_expectation": "LLM generated expectation",
        "code_expectation": 42,
    }

    result = extract_expectations_from_trace(trace, source_type="HUMAN")
    assert result == {"human_expectation": {"expected": "Answer from human"}}

    result = extract_expectations_from_trace(trace, source_type="LLM_JUDGE")
    assert result == {"llm_expectation": "LLM generated expectation"}

    result = extract_expectations_from_trace(trace, source_type="CODE")
    assert result == {"code_expectation": 42}

    result = extract_expectations_from_trace(trace, source_type="human")
    assert result == {"human_expectation": {"expected": "Answer from human"}}

    with pytest.raises(mlflow.exceptions.MlflowException, match="Invalid assessment source type"):
        extract_expectations_from_trace(trace, source_type="INVALID_SOURCE")

def test_convert_predict_fn_async_function():
    async def async_predict_fn(request):
        await asyncio.sleep(0.01)
        return "async test response"

    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    converted_fn = convert_predict_fn(async_predict_fn, sample_input)

    result = converted_fn(request=sample_input)
    assert result == "async test response"

    traces = get_traces()
    assert len(traces) == 1
    purge_traces()

def test_evaluate_with_async_predict_fn():
    async def async_predict_fn(request):
        await asyncio.sleep(0.01)
        return "async test response"

    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    @scorer
    def dummy_scorer(inputs, outputs):
        return 0

    mlflow.genai.evaluate(
        data=[{"inputs": sample_input}],
        predict_fn=async_predict_fn,
        scorers=[dummy_scorer],
    )
    assert len(get_traces()) == 1
    purge_traces()


# --- tests/langchain/test_langchain_autolog.py ---

def test_autolog_record_exception(async_logging_enabled):
    def always_fail(input):
        raise Exception("Error!")

    model = RunnableLambda(always_fail)

    mlflow.langchain.autolog()

    with pytest.raises(Exception, match="Error!"):
        model.invoke("test")

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "always_fail"

def test_chat_model_autolog():
    mlflow.langchain.autolog()
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the weather in San Francisco?"),
        AIMessage(
            content="foo",
            tool_calls=[{"name": "GetWeather", "args": {"location": "San Francisco"}, "id": "123"}],
        ),
        ToolMessage(content="Weather in San Francisco is 70F.", tool_call_id="123"),
    ]
    response = model.invoke(messages)

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "ChatOpenAI"
    assert span.span_type == "CHAT_MODEL"
    _LC_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    for msg, expected in zip(span.inputs["messages"], messages, strict=True):
        assert msg["role"] == _LC_TYPE_TO_ROLE[expected.type]
        assert msg["content"] == expected.content
    assert span.outputs["choices"][0]["message"]["content"] == response.content
    assert span.get_attribute("invocation_params")["model"] == "gpt-4o-mini"
    assert span.get_attribute("invocation_params")["temperature"] == 0.9
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "langchain"
    assert span.model_name == "gpt-4o-mini"

def test_chat_model_autolog_audio_input_normalization(mime_type, expected_format):
    audio_b64 = "SGVsbG8="

    class AudioInputModel(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="heard it"))])

        @property
        def _llm_type(self):
            return "audio-input-model"

    mlflow.langchain.autolog()
    model = AudioInputModel()
    model.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": "What is this?"},
                {
                    "type": "audio",
                    "source_type": "base64",
                    "data": audio_b64,
                    "mime_type": mime_type,
                },
            ]
        )
    ])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    span = next(s for s in trace.data.spans if s.span_type == "CHAT_MODEL")

    msgs = span.inputs["messages"]
    audio_block = msgs[0]["content"][1]
    assert audio_block["type"] == "input_audio"
    assert audio_block["input_audio"]["format"] == expected_format
    attachment_uri = audio_block["input_audio"]["data"]
    assert attachment_uri.startswith("mlflow-attachment://")
    expected_mime = "mpeg" if expected_format == "mp3" else expected_format
    assert f"content_type=audio%2F{expected_mime}" in attachment_uri

def test_chat_model_autolog_audio_output_normalization():
    audio_b64 = "SGVsbG8="

    class AudioOutputModel(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            ai_msg = AIMessage(
                content=[
                    {"type": "text", "text": "Here is audio."},
                    {
                        "type": "audio",
                        "source_type": "base64",
                        "data": audio_b64,
                        "mime_type": "audio/wav",
                    },
                ]
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        @property
        def _llm_type(self):
            return "audio-output-model"

    mlflow.langchain.autolog()
    model = AudioOutputModel()
    model.invoke([("human", "Give me audio")])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    span = next(s for s in trace.data.spans if s.span_type == "CHAT_MODEL")

    audio_block = span.outputs["choices"][0]["message"]["content"][1]
    assert audio_block["type"] == "input_audio"
    assert audio_block["input_audio"]["format"] == "wav"
    attachment_uri = audio_block["input_audio"]["data"]
    assert attachment_uri.startswith("mlflow-attachment://")
    assert "content_type=audio%2Fwav" in attachment_uri

def test_chat_model_autolog_openai_audio_output_with_format():
    audio_b64 = "SGVsbG8="

    class OpenAIAudioModelWithFormat(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            ai_msg = AIMessage(
                content="",
                additional_kwargs={
                    "audio": {
                        "id": "audio_abc123",
                        "data": audio_b64,
                        "expires_at": 9999999999,
                        "transcript": "Yes, I am.",
                    }
                },
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        @property
        def _llm_type(self):
            return "openai-audio-model"

        @property
        def _identifying_params(self):
            return {
                "model": "gpt-4o-audio-preview",
                "audio": {"voice": "alloy", "format": "wav"},
            }

    mlflow.langchain.autolog()
    model = OpenAIAudioModelWithFormat()
    model.invoke([("human", "Are you an AI?")])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    span = next(s for s in trace.data.spans if s.span_type == "CHAT_MODEL")

    content = span.outputs["choices"][0]["message"]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Yes, I am."}
    assert content[1]["type"] == "input_audio"
    attachment_uri = content[1]["input_audio"]["data"]
    assert attachment_uri.startswith("mlflow-attachment://")
    assert "content_type=audio%2Fwav" in attachment_uri
    assert content[1]["input_audio"]["format"] == "wav"

def test_chat_model_autolog_openai_audio_transcript_fallback():

    class OpenAIAudioModel(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            ai_msg = AIMessage(
                content="",
                additional_kwargs={
                    "audio": {
                        "id": "audio_abc123",
                        "data": "SGVsbG8=",
                        "expires_at": 9999999999,
                        "transcript": "Yes, I am.",
                    }
                },
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        @property
        def _llm_type(self):
            return "openai-audio-model"

    mlflow.langchain.autolog()
    model = OpenAIAudioModel()
    model.invoke([("human", "Are you an AI?")])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    span = next(s for s in trace.data.spans if s.span_type == "CHAT_MODEL")

    assert span.outputs["choices"][0]["message"]["content"] == "Yes, I am."

def test_chat_model_autolog_openai_audio_transcript_no_override():
    class AudioModelWithContent(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            ai_msg = AIMessage(
                content="I have text content.",
                additional_kwargs={
                    "audio": {
                        "id": "audio_abc123",
                        "data": "SGVsbG8=",
                        "expires_at": 9999999999,
                        "transcript": "Different transcript.",
                    }
                },
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        @property
        def _llm_type(self):
            return "audio-model-with-content"

    mlflow.langchain.autolog()
    model = AudioModelWithContent()
    model.invoke([("human", "Say something")])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    span = next(s for s in trace.data.spans if s.span_type == "CHAT_MODEL")

    assert span.outputs["choices"][0]["message"]["content"] == "I have text content."

def test_chat_model_bind_tool_autolog():
    mlflow.langchain.autolog()

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather in {location} is 70F."

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    model_with_tools = model.bind_tools([get_weather])
    model_with_tools.invoke("What is the weather in San Francisco?")

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "ChatOpenAI"
    assert span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location.",
                "parameters": {
                    "properties": {
                        "location": {
                            "type": "string",
                        }
                    },
                    "required": ["location"],
                    "type": "object",
                },
            },
        }
    ]
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "langchain"
    assert span.model_name == "gpt-4o-mini"

def test_retriever_autolog(tmp_path, async_logging_enabled):
    mlflow.langchain.autolog()
    model, query = create_retriever(tmp_path)
    model.invoke(query)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "VectorStoreRetriever"
    assert spans[0].span_type == "RETRIEVER"
    assert spans[0].inputs == query
    assert spans[0].outputs[0]["metadata"] == {"source": "tests/langchain/state_of_the_union.txt"}

def test_langchain_autolog_callback_injection_in_invoke(invoke_arg, config, async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        model.invoke(input, config)
    elif invoke_arg == "kwargs":
        model.invoke(input, config=config)
    elif invoke_arg is None:
        model.invoke(input)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == [{"role": "user", "content": "What is MLflow?"}]
    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        # NB: Langchain has a bug that the callback is called different times when
        # passed by a list or a callback manager. As a workaround we only check
        # the content of the events not the count.
        # https://github.com/langchain-ai/langchain/issues/24642
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}

async def test_langchain_autolog_callback_injection_in_ainvoke(
    invoke_arg, config, async_logging_enabled
):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        await model.ainvoke(input, config)
    elif invoke_arg == "kwargs":
        await model.ainvoke(input, config=config)
    elif invoke_arg is None:
        await model.ainvoke(input)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == [{"role": "user", "content": "What is MLflow?"}]

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        # NB: Langchain has a bug that the callback is called different times when
        # passed by a list or a callback manager. As a workaround we only check
        # the content of the events not the count.
        # https://github.com/langchain-ai/langchain/issues/24642
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}

def test_langchain_autolog_callback_injection_in_batch(invoke_arg, config, async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        model.batch([input] * 2, config)
    elif invoke_arg == "kwargs":
        model.batch([input] * 2, config=config)
    elif invoke_arg is None:
        model.batch([input] * 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.info.status == "OK"
        assert trace.data.spans[0].name == "RunnableSequence"
        assert trace.data.spans[0].inputs == input
        assert trace.data.spans[0].outputs == [{"role": "user", "content": "What is MLflow?"}]

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        for handler in handlers:
            assert set(handler.logs) == {"chain_start", "chain_end"}

def test_tracing_source_run_in_batch():
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    input = {"product": "MLflow"}
    with mlflow.start_run() as run:
        model.batch([input] * 2)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run.info.run_id

def test_tracing_source_run_in_pyfunc_model_predict(model_info):
    mlflow.langchain.autolog()

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with mlflow.start_run() as run:
        pyfunc_model.predict([{"product": "MLflow"}] * 2)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run.info.run_id

async def test_langchain_autolog_callback_injection_in_abatch(
    invoke_arg, config, async_logging_enabled
):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        await model.abatch([input] * 2, config)
    elif invoke_arg == "kwargs":
        await model.abatch([input] * 2, config=config)
    elif invoke_arg is None:
        await model.abatch([input] * 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.info.status == "OK"
        assert trace.data.spans[0].name == "RunnableSequence"
        assert trace.data.spans[0].inputs == input
        assert trace.data.spans[0].outputs == [{"role": "user", "content": "What is MLflow?"}]

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        for handler in handlers:
            assert set(handler.logs) == {"chain_start", "chain_end"}

def test_langchain_autolog_callback_injection_in_stream(invoke_arg, config, async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)

    input = {"product": "MLflow"}
    if invoke_arg == "args":
        list(model.stream(input, config))
    elif invoke_arg == "kwargs":
        list(model.stream(input, config=config))
    elif invoke_arg is None:
        list(model.stream(input))

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == "Hello world"

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}

async def test_langchain_autolog_callback_injection_in_astream(
    invoke_arg, config, async_logging_enabled
):
    mlflow.langchain.autolog()

    model = create_openai_runnable()
    original_handlers = _extract_callback_handlers(config)
    _reset_callback_handlers(original_handlers)
    input = {"product": "MLflow"}

    async def invoke_astream(model, config):
        if invoke_arg == "args":
            astream = model.astream(input, config)
        elif invoke_arg == "kwargs":
            astream = model.astream(input, config=config)
        elif invoke_arg is None:
            astream = model.astream(input)

        # Consume the stream
        async for _ in astream:
            pass

    await invoke_astream(model, config)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].name == "RunnableSequence"
    assert traces[0].data.spans[0].inputs == input
    assert traces[0].data.spans[0].outputs == "Hello world"

    # Original callback should not be mutated
    handlers = _extract_callback_handlers(config)
    assert handlers == original_handlers

    # The original callback is called by the chain
    if handlers and invoke_arg:
        assert set(handlers[0].logs) == {"chain_start", "chain_end"}

def test_langchain_autolog_tracing_thread_safe(async_logging_enabled):
    mlflow.langchain.autolog()

    model = create_openai_runnable()

    def _invoke():
        # Add random sleep to simulate real LLM prediction
        time.sleep(random.uniform(0.1, 0.5))

        model.invoke({"product": "MLflow"})

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="test-langchain-autolog") as executor:
        futures = [executor.submit(_invoke) for _ in range(30)]
        _ = [f.result() for f in futures]

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 30
    for trace in traces:
        assert trace.info.status == "OK"
        assert len(trace.data.spans) == 4
        assert trace.data.spans[0].name == "RunnableSequence"

def test_langchain_tracer_injection_for_arbitrary_runnables(log_traces, async_logging_enabled):
    should_log_traces = log_traces is not False

    if log_traces is not None:
        mlflow.langchain.autolog(log_traces=log_traces)
    else:
        mlflow.langchain.autolog()

    add = RunnableLambda(func=lambda x: x + 1)
    square = RunnableLambda(func=lambda x: x**2)
    model = RouterRunnable(runnables={"add": add, "square": square})

    model.invoke({"key": "square", "input": 3})

    if async_logging_enabled and should_log_traces:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    if should_log_traces:
        assert len(traces) == 1
        assert traces[0].data.spans[0].span_type == "CHAIN"
    else:
        assert len(traces) == 0

def test_set_retriever_schema_work_for_langchain_model(model_info):
    from mlflow.models.dependencies_schemas import DependenciesSchemasType, set_retriever_schema

    set_retriever_schema(
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )

    mlflow.langchain.autolog()

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    pyfunc_model.predict("MLflow")

    traces = get_traces()
    assert len(traces) == 1
    assert DependenciesSchemasType.RETRIEVERS.value in traces[0].info.tags

    purge_traces()

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    list(pyfunc_model.predict_stream("MLflow"))

    traces = get_traces()
    assert len(traces) == 1
    assert DependenciesSchemasType.RETRIEVERS.value in traces[0].info.tags

def test_langchain_auto_tracing_in_serving_agent():
    mlflow.langchain.autolog()

    input_example = {"input": "What is 2 * 3?"}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            "tests/langchain/sample_code/openai_agent.py",
            name="langchain_model",
            input_example=input_example,
        )

    databricks_request_id, response, trace_dict = score_in_model_serving(
        model_info.model_uri,
        input_example,
    )

    trace = Trace.from_dict(trace_dict)
    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.client_request_id == databricks_request_id
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 7

    root_span = spans[0]
    assert root_span.name == "LangGraph"
    assert root_span.span_type == SpanType.CHAIN
    assert root_span.inputs["input"] == "What is 2 * 3?"
    assert root_span.outputs["messages"][-1]["content"] == "The result of 2 * 3 is 6."
    assert root_span.start_time_ns // 1_000_000 == trace.info.timestamp_ms
    assert (
        root_span.end_time_ns // 1_000_000
        - (trace.info.timestamp_ms + trace.info.execution_time_ms)
    ) <= 1

def test_langchain_tracing_multi_threads():
    mlflow.langchain.autolog()

    temperatures = [(t + 1) / 10 for t in range(4)]
    models = [create_openai_runnable(temperature=t) for t in temperatures]

    with ThreadPoolExecutor(
        max_workers=len(temperatures), thread_name_prefix="test-langchain-concurrent"
    ) as executor:
        futures = [executor.submit(models[i].invoke, {"product": "MLflow"}) for i in range(4)]
        for f in futures:
            f.result()

    traces = get_traces()
    assert len(traces) == 4
    assert (
        sorted(
            trace.data.spans[2].get_attribute("invocation_params")["temperature"]
            for trace in traces
        )
        == temperatures
    )

def test_autolog_link_traces_to_loaded_model(model_infos, func):
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        msg = {"product": f"{loaded_model.steps[1].temperature}_{model_info.model_id}"}
        if func == "invoke":
            loaded_model.invoke(msg)
        elif func == "batch":
            loaded_model.batch([msg])
        elif func == "stream":
            list(loaded_model.stream(msg))

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        temp = trace.data.spans[2].get_attribute("invocation_params")["temperature"]
        logged_temp, logged_model_id = json.loads(trace.data.request)["product"].split(
            "_", maxsplit=1
        )
        assert logged_model_id is not None
        assert str(temp) == logged_temp
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == logged_model_id

async def test_autolog_link_traces_to_loaded_model_async(model_infos, func):
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.langchain.load_model(model_info.model_uri)
        msg = {"product": f"{loaded_model.steps[1].temperature}_{model_info.model_id}"}
        if func == "ainvoke":
            await loaded_model.ainvoke(msg)
        elif func == "abatch":
            await loaded_model.abatch([msg])
        elif func == "astream":
            async for chunk in loaded_model.astream(msg):
                pass

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        temp = trace.data.spans[2].get_attribute("invocation_params")["temperature"]
        logged_temp, logged_model_id = json.loads(trace.data.request)["product"].split(
            "_", maxsplit=1
        )
        assert logged_model_id is not None
        assert str(temp) == logged_temp
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == logged_model_id

def test_autolog_link_traces_to_loaded_model_pyfunc(model_infos):
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        loaded_model.predict({"product": model_info.model_id})

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        logged_model_id = json.loads(trace.data.request)["product"]
        assert logged_model_id is not None
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == logged_model_id

def test_autolog_link_traces_to_active_model(model_infos):
    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(model_id=model.model_id)
    mlflow.langchain.autolog()

    for model_info in model_infos:
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        loaded_model.predict({"product": model_info.model_id})

    traces = get_traces()
    assert len(traces) == len(model_infos)
    for trace in traces:
        logged_model_id = json.loads(trace.data.request)["product"]
        assert logged_model_id is not None
        assert trace.info.request_metadata[TraceMetadataKey.MODEL_ID] == model.model_id
        assert model.model_id != logged_model_id

def test_langchain_tracing_evaluate(log_traces):
    from mlflow.genai import scorer

    if log_traces:
        mlflow.langchain.autolog()
        mlflow.openai.autolog()  # Our chain contains OpenAI call as well

    chain = create_openai_runnable()

    data = [
        {
            "inputs": {"product": "MLflow"},
            "expectations": {"expected_response": "MLflow is an open-source platform."},
        },
        {
            "inputs": {"product": "Spark"},
            "expectations": {"expected_response": "Spark is a unified analytics engine."},
        },
    ]

    def predict_fn(product: str) -> str:
        return chain.invoke({"product": product})

    @scorer
    def exact_match(outputs: str, expectations: dict[str, str]) -> bool:
        return outputs == expectations["expected_response"]

    result = mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=data,
        scorers=[exact_match],
    )
    assert result.metrics["exact_match/mean"] == 0.0
    assert result.result_df is not None

    # Traces should be enabled automatically
    assert len(get_traces()) == 2
    for trace in get_traces():
        assert len(trace.data.spans) == 5
        assert trace.data.spans[0].name == "RunnableSequence"
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == result.run_id
        assert len(trace.info.assessments) == 2

async def test_autolog_run_tracer_inline_with_manual_traces_async():
    mlflow.langchain.autolog(run_tracer_inline=True)

    prompt = PromptTemplate(
        input_variables=["color"],
        template="What is the complementary color of {color}?",
    )
    llm = ChatOpenAI()

    @mlflow.trace
    def manual_transform(s: str):
        return s.replace("red", "blue")

    chain = RunnableLambda(manual_transform) | prompt | llm | StrOutputParser()

    @mlflow.trace(name="parent")
    async def run(message):
        return await chain.ainvoke(message)

    response = await run("red")
    expected_response = '[{"role": "user", "content": "What is the complementary color of blue?"}]'
    assert response == expected_response

    traces = get_traces()
    assert len(traces) == 1

    trace = traces[0]
    spans = trace.data.spans
    assert spans[0].name == "parent"
    assert spans[1].name == "RunnableSequence"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "manual_transform"
    assert spans[2].parent_id == spans[1].span_id
    # Find and verify ChatOpenAI span has model name
    chat_model_span = next(s for s in spans if s.name == "ChatOpenAI")
    assert chat_model_span.model_name == "gpt-3.5-turbo"


# --- tests/langchain/test_langchain_databricks_integration.py ---

def test_save_and_load_chat_databricks(model_path):
    from databricks_langchain import ChatDatabricks

    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
    prompt = PromptTemplate.from_template("What is {product}?")
    chain = prompt | llm | StrOutputParser()

    mlflow.langchain.save_model(chain, path=model_path)

    loaded_model = mlflow.langchain.load_model(model_path)
    assert loaded_model == chain

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_path)
    prediction = loaded_pyfunc_model.predict([{"product": "MLflow"}])
    assert prediction == ["What is MLflow?"]


# --- tests/langchain/test_langchain_model_export.py ---

def test_langchain_native_log_and_load_model():
    model = create_openai_runnable()

    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            model, name="langchain_model", input_example={"product": "MLflow"}
        )

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string (required)]"
    assert str(logged_model.signature.outputs) == "[string (required)]"

    assert type(loaded_model) == RunnableSequence
    assert loaded_model.steps[0].template == "What is {product}?"
    assert type(loaded_model.steps[1]).__name__ == "ChatOpenAI"

    # Predict
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    result = loaded_model.predict([{"product": "MLflow"}])
    assert result == [json.dumps(TEST_CONTENT)]

    # Predict stream
    result = loaded_model.predict_stream([{"product": "MLflow"}])
    assert inspect.isgenerator(result)
    assert list(result) == ["Hello", " world"]

def test_pyfunc_spark_udf_with_langchain_model(spark):
    model = create_openai_runnable()
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(
            model, name="langchain_model", input_example={"product": "MLflow"}
        )
    loaded_model = mlflow.pyfunc.spark_udf(spark, logged_model.model_uri, result_type="string")
    df = spark.createDataFrame([("MLflow",), ("Spark",)], ["product"])
    df = df.withColumn("answer", loaded_model())
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [
        '[{"role": "user", "content": "What is MLflow?"}]',
        '[{"role": "user", "content": "What is Spark?"}]',
    ]

def test_agent_with_unpicklable_tools(tmp_path):
    from langchain.agents import AgentType, initialize_agent

    tmp_file = tmp_path / "temp_file.txt"
    with open(tmp_file, mode="w") as temp_file:
        # files that aren't opened for reading cannot be pickled
        tools = [
            Tool.from_function(
                func=lambda: temp_file,
                name="Write 0",
                description="If you need to write 0 to a file",
            )
        ]
        agent = initialize_agent(
            llm=OpenAI(temperature=0),
            tools=tools,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        with pytest.raises(
            MlflowException,
            match=(
                "Error when attempting to pickle the AgentExecutor tools. "
                "This model likely does not support serialization."
            ),
        ):
            with mlflow.start_run():
                mlflow.langchain.log_model(agent, name="unpicklable_tools")

def test_save_load_runnable_passthrough():
    runnable = RunnablePassthrough()
    assert runnable.invoke("hello") == "hello"

    input_example = "hello"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, name="model_path", input_example=input_example
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(input_example) == "hello"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(["hello"]) == ["hello"]

def test_save_load_runnable_lambda(spark):
    def add_one(x: int) -> int:
        return x + 1

    runnable = RunnableLambda(add_one)

    assert runnable.invoke(1) == 2
    assert runnable.batch([1, 2, 3]) == [2, 3, 4]

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, name="runnable_lambda", input_example=[1, 2, 3]
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(1) == 2
    assert loaded_model.batch([1, 2, 3]) == [2, 3, 4]

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert loaded_model.predict(1) == [2]
    assert loaded_model.predict([1, 2, 3]) == [2, 3, 4]

    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, result_type="long")
    df = spark.createDataFrame([(1,), (2,), (3,)], ["data"])
    df = df.withColumn("answer", udf("data"))
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [2, 3, 4]

def test_save_load_runnable_lambda_in_sequence():
    def add_one(x):
        return x + 1

    def mul_two(x):
        return x * 2

    runnable_1 = RunnableLambda(add_one)
    runnable_2 = RunnableLambda(mul_two)
    sequence = runnable_1 | runnable_2
    assert sequence.invoke(1) == 4

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            sequence, name="model_path", input_example=[1, 2, 3]
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke(1) == 4
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict(1) == [4]
    assert pyfunc_loaded_model.predict([1, 2, 3]) == [4, 6, 8]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [4, 6, 8]
    }

def test_save_load_runnable_parallel():
    runnable = RunnableParallel({"llm": create_openai_runnable()})
    expected_result = {"llm": json.dumps(TEST_CONTENT)}
    assert runnable.invoke({"product": "MLflow"}) == expected_result
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            runnable, name="model_path", input_example=[{"product": "MLflow"}]
        )
    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke({"product": "MLflow"}) == expected_result
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict([{"product": "MLflow"}]) == [expected_result]

    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": [expected_result]
    }

def test_runnable_branch_save_load():
    branch = RunnableBranch(
        (lambda x: isinstance(x, str), lambda x: x.upper()),
        (lambda x: isinstance(x, int), lambda x: x + 1),
        (lambda x: isinstance(x, float), lambda x: x * 2),
        lambda x: "goodbye",
    )

    assert branch.invoke("hello") == "HELLO"
    assert branch.invoke({}) == "goodbye"

    with mlflow.start_run():
        # We only support single input format for now, so we should
        # not save signature for runnable branch which accepts multiple
        # input types
        model_info = mlflow.langchain.log_model(branch, name="model_path")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == "HELLO"
    assert loaded_model.invoke({}) == "goodbye"
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == "HELLO"
    assert pyfunc_loaded_model.predict({}) == "goodbye"

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": "hello"}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert PredictionsResponse.from_json(response.content.decode("utf-8")) == {
        "predictions": "HELLO"
    }

def test_pyfunc_builtin_chat_request_conversion_fails_gracefully():
    chain = RunnablePassthrough() | itemgetter("messages")
    # Ensure we're going to test that "messages" remains intact & unchanged even if it
    # doesn't appear explicitly in the chain's input schema
    assert "messages" not in chain.input_schema().model_fields

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, name="model_path")
        pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    assert pyfunc_loaded_model.predict({"messages": "not an array"}) == "not an array"

    # Verify that messages aren't converted to LangChain format if extra keys are present,
    # under the assumption that additional keys can't be specified when calling LangChain invoke()
    # / batch() with chat messages
    assert pyfunc_loaded_model.predict({
        "messages": [{"role": "user", "content": "blah"}],
        "extrakey": "extra",
    }) == [
        {"role": "user", "content": "blah"},
    ]

    # Verify that messages aren't converted to LangChain format if role / content are missing
    # or extra keys are present in the message
    assert pyfunc_loaded_model.predict({
        "messages": [{"content": "blah"}],
    }) == [
        {"content": "blah"},
    ]
    assert pyfunc_loaded_model.predict({
        "messages": [{"role": "user", "content": "blah"}, {}],
    }) == [
        {"role": "user", "content": "blah"},
        {},
    ]
    assert pyfunc_loaded_model.predict({
        "messages": [{"role": "user", "content": 123}],
    }) == [
        {"role": "user", "content": 123},
    ]

    # Verify behavior for batches of message histories
    assert pyfunc_loaded_model.predict([
        {
            "messages": "not an array",
        },
        {
            "messages": [{"role": "user", "content": "content"}],
        },
    ]) == [
        "not an array",
        [{"role": "user", "content": "content"}],
    ]
    assert pyfunc_loaded_model.predict([
        {
            "messages": [{"role": "user", "content": "content"}],
        },
        {"messages": [{"role": "user", "content": "content"}], "extrakey": "extra"},
    ]) == [
        [{"role": "user", "content": "content"}],
        [{"role": "user", "content": "content"}],
    ]
    assert pyfunc_loaded_model.predict([
        {
            "messages": [{"role": "user", "content": "content"}],
        },
        {
            "messages": [
                {"role": "user", "content": "content"},
                {"role": "user", "content": 123},
            ],
        },
    ]) == [
        [{"role": "user", "content": "content"}],
        [{"role": "user", "content": "content"}, {"role": "user", "content": 123}],
    ]

def test_save_load_chain_as_code_model_config_dict(chain_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name="model_path",
            input_example=input_example,
            model_config={
                "response": "modified response",
                "embedding_size": 5,
                "llm_prompt_template": "answer the question",
            },
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    answer = "modified response"
    assert loaded_model.invoke(input_example) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert answer == _get_message_content(pyfunc_loaded_model.predict(input_example))

def test_save_load_chain_as_code_with_different_names(tmp_path, model_config):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }

    # Read the contents of the original chain file
    with open("tests/langchain/sample_code/chain.py") as chain_file:
        chain_file_content = chain_file.read()

    temp_file = tmp_path / "model.py"
    temp_file.write_text(chain_file_content)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            str(temp_file),
            name="model_path",
            input_example=input_example,
            model_config=model_config,
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    answer = "Databricks"
    assert loaded_model.invoke(input_example) == answer
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert answer == _get_message_content(pyfunc_loaded_model.predict(input_example))

def test_save_load_chain_as_code_multiple_times(tmp_path, chain_path, model_config):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name="model_path",
            input_example=input_example,
            model_config=model_config,
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    with open(model_config) as f:
        base_config = yaml.safe_load(f)

    assert loaded_model.middle[0].messages[0].prompt.template == base_config["llm_prompt_template"]

    file_name = "config_updated.yml"
    new_config_file = str(tmp_path.joinpath(file_name))

    new_config = base_config.copy()
    new_config["llm_prompt_template"] = "new_template"
    with open(new_config_file, "w") as f:
        yaml.dump(new_config, f)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name="model_path",
            input_example=input_example,
            model_config=new_config_file,
        )

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.middle[0].messages[0].prompt.template == new_config["llm_prompt_template"]

def test_save_load_chain_errors(chain_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    with mlflow.start_run():
        with pytest.raises(
            MlflowException,
            match=f"The provided model path '{chain_path}' does not exist. "
            "Ensure the file path is valid and try again.",
        ):
            mlflow.langchain.log_model(
                chain_path,
                name="model_path",
                input_example=input_example,
                model_config="tests/langchain/state_of_the_union.txt",
            )

def test_save_load_langchain_binding_llm_with_tool():
    from langchain_core.tools import tool

    # We need to use ChatOpenAI from langchain_openai as community one does not support bind_tools
    from langchain_openai import ChatOpenAI

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b.

        Args:
            a: first int
            b: second int
        """
        return a + b

    runnable_binding = ChatOpenAI(temperature=0.9).bind_tools([add])
    model = runnable_binding | StrOutputParser()
    expected_output = '[{"role": "user", "content": "hello"}]'
    assert model.invoke("hello") == expected_output

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(model, name="model_path", input_example="hello")

    loaded_model = mlflow.langchain.load_model(model_info.model_uri)
    assert loaded_model.invoke("hello") == expected_output
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_loaded_model.predict("hello") == [expected_output]

def test_agent_executor_model_with_messages_input():
    question = {"messages": [{"role": "user", "content": "Who owns MLflow?"}]}

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            os.path.abspath("tests/langchain/agent_executor/chain.py"),
            name="model_path",
            input_example=question,
            model_config=os.path.abspath("tests/langchain/agent_executor/config.yml"),
        )
    native_model = mlflow.langchain.load_model(model_info.model_uri)
    assert native_model.invoke(question)["output"] == "Databricks"
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    # TODO: in the future we should fix this and output shouldn't be wrapped
    # The result is wrapped in a list because during signature enforcement we convert
    # input data to pandas dataframe, then inside _convert_llm_input_data
    # we convert pandas dataframe back to records, and a single row will be
    # wrapped inside a list.
    assert pyfunc_model.predict(question) == ["Databricks"]

    # Test stream output
    response = pyfunc_model.predict_stream(question)
    assert inspect.isgenerator(response)

    expected_response = [
        {
            "output": "Databricks",
            "messages": [
                {
                    "additional_kwargs": {},
                    "content": "Databricks",
                    "example": False,
                    "id": None,
                    "invalid_tool_calls": [],
                    "name": None,
                    "response_metadata": {},
                    "tool_calls": [],
                    "type": "ai",
                    "usage_metadata": None,
                }
            ],
        }
    ]
    assert list(response) == expected_response

def test_custom_resources(tmp_path):
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is a good name for a company that makes MLflow?",
            }
        ]
    }
    expected_resources = {
        "api_version": "1",
        "databricks": {
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-bge-large-en"},
                {"name": "azure-eastus-model-serving-2_vs_endpoint"},
            ],
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
            "sql_warehouse": [{"name": "testid"}],
            "function": [
                {"name": "rag.studio.test_function_a"},
                {"name": "rag.studio.test_function_b"},
            ],
        },
    }
    artifact_path = "model_path"
    chain_path = "tests/langchain/sample_code/chain.py"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path,
            input_example=input_example,
            model_config="tests/langchain/sample_code/config.yml",
            resources=[
                DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
                DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
                DatabricksServingEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
                DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
                DatabricksSQLWarehouse(warehouse_id="testid"),
                DatabricksFunction(function_name="rag.studio.test_function_a"),
                DatabricksFunction(function_name="rag.studio.test_function_b"),
            ],
        )

        model_path = _download_artifact_from_uri(model_info.model_uri)
        reloaded_model = Model.load(os.path.join(model_path, "MLmodel"))
        assert reloaded_model.resources == expected_resources

    yaml_file = tmp_path.joinpath("resources.yaml")
    with open(yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                - name: databricks-bge-large-en
                - name: azure-eastus-model-serving-2_vs_endpoint
                sql_warehouse:
                - name: testid
                function:
                - name: rag.studio.test_function_a
                - name: rag.studio.test_function_b
            """
        )

    artifact_path_2 = "model_path_2"
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            chain_path,
            name=artifact_path_2,
            input_example=input_example,
            model_config="tests/langchain/sample_code/config.yml",
            resources=yaml_file,
        )

        model_path = _download_artifact_from_uri(model_info.model_uri)
        reloaded_model = Model.load(os.path.join(model_path, "MLmodel"))
        assert reloaded_model.resources == expected_resources

def test_pyfunc_converts_chat_request_for_non_chat_model():
    input_example = {"messages": [{"role": "user", "content": "Hello"}]}
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=SIMPLE_MODEL_CODE_PATH,
            input_example=input_example,
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    result = pyfunc_model.predict(input_example)
    # output are converted to chatResponse format
    assert isinstance(result[0]["choices"][0]["message"]["content"], str)

    response = pyfunc_model.predict_stream(input_example)
    assert inspect.isgenerator(response)
    assert isinstance(list(response)[0]["choices"][0]["delta"]["content"], str)

def test_langchain_v1_save_model_as_pickle_error():
    model = create_openai_runnable()
    with mlflow.start_run():
        with pytest.raises(
            MlflowException,
            match="LangChain v1 onward only supports models-from-code",
        ):
            mlflow.langchain.log_model(
                model, name="langchain_model", input_example={"product": "MLflow"}
            )


# --- tests/langchain/test_langchain_tracer.py ---

def test_llm_success():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_llm_start(
        {},
        ["test prompt"],
        run_id=run_id,
        name="test_llm",
    )

    callback.on_llm_new_token("test", run_id=run_id)

    callback.on_llm_end(LLMResult(generations=[[{"text": "generated text"}]]), run_id=run_id)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    llm_span = trace.data.spans[0]

    assert llm_span.name == "test_llm"

    assert llm_span.span_type == "LLM"
    assert llm_span.start_time_ns is not None
    assert llm_span.end_time_ns is not None
    assert llm_span.status == SpanStatus(SpanStatusCode.OK)
    assert llm_span.inputs == ["test prompt"]
    assert llm_span.outputs["choices"][0]["message"]["content"] == "generated text"
    assert llm_span.events[0].name == "new_token"

    _validate_trace_json_serialization(trace)

def test_llm_internal_exception():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_llm_start(
        {},
        ["test prompt"],
        run_id=run_id,
        name="test_llm",
    )
    try:
        with pytest.raises(
            Exception,
            match="Span for run_id dummy not found.",
        ):
            callback.on_llm_end(LLMResult(generations=[[{"text": "generated"}]]), run_id="dummy")
    finally:
        callback.flush()

def test_chat_model():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    input_messages = [SystemMessage("system prompt"), HumanMessage("test prompt")]
    callback.on_chat_model_start(
        {},
        [input_messages],
        run_id=run_id,
        name="test_chat_model",
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "generated text"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    chat_model_span = trace.data.spans[0]
    assert chat_model_span.name == "test_chat_model"
    assert chat_model_span.span_type == "CHAT_MODEL"
    assert chat_model_span.status.status_code == SpanStatusCode.OK
    assert chat_model_span.inputs["messages"][0]["role"] == "system"
    assert chat_model_span.inputs["messages"][0]["content"] == "system prompt"
    assert chat_model_span.inputs["messages"][1]["role"] == "user"
    assert chat_model_span.inputs["messages"][1]["content"] == "test prompt"
    assert chat_model_span.outputs["choices"][0]["message"]["content"] == "generated text"

def test_chat_model_with_tool():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    input_messages = [HumanMessage("test prompt")]
    # OpenAI tool format
    tool_definition = {
        "type": "function",
        "function": {
            "name": "GetWeather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "properties": {
                    "location": {
                        "description": "The city and state, e.g. San Francisco, CA",
                        "type": "string",
                    }
                },
                "required": ["location"],
                "type": "object",
            },
        },
    }
    callback.on_chat_model_start(
        {},
        [input_messages],
        run_id=run_id,
        name="test_chat_model",
        invocation_params={"tools": [tool_definition]},
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "generated text"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    chat_model_span = trace.data.spans[0]
    assert chat_model_span.status.status_code == SpanStatusCode.OK
    assert chat_model_span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == [tool_definition]

def test_chat_model_with_non_openai_tool():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    input_messages = [HumanMessage("test prompt")]
    # Anthropic tool format
    tool_definition = {
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "input_schema": {
            "properties": {
                "location": {
                    "description": "The city and state, e.g. San Francisco, CA",
                    "type": "string",
                }
            },
            "required": ["location"],
            "type": "object",
        },
    }
    callback.on_chat_model_start(
        {},
        [input_messages],
        run_id=run_id,
        name="test_chat_model",
        invocation_params={"tools": [tool_definition]},
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "generated text"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    chat_model_span = trace.data.spans[0]
    assert chat_model_span.status.status_code == SpanStatusCode.OK
    assert chat_model_span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location.",
            },
        }
    ]

def test_retriever_success():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_retriever_start(
        {},
        query="test query",
        run_id=run_id,
        name="test_retriever",
    )

    documents = [
        Document(
            page_content="document content 1",
            metadata={"chunk_id": "1", "doc_uri": "uri1"},
        ),
        Document(
            page_content="document content 2",
            metadata={"chunk_id": "2", "doc_uri": "uri2"},
        ),
    ]
    callback.on_retriever_end(documents, run_id=run_id)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    retriever_span = trace.data.spans[0]

    assert retriever_span.name == "test_retriever"
    assert retriever_span.span_type == "RETRIEVER"
    assert retriever_span.inputs == "test query"
    assert retriever_span.outputs == [
        MlflowDocument.from_langchain_document(doc).to_dict() for doc in documents
    ]
    assert retriever_span.start_time_ns is not None
    assert retriever_span.end_time_ns is not None
    assert retriever_span.status.status_code == SpanStatusCode.OK

    _validate_trace_json_serialization(trace)

def test_retriever_internal_exception():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_retriever_start(
        {},
        query="test query",
        run_id=run_id,
        name="test_retriever",
    )

    try:
        with pytest.raises(
            Exception,
            match="Span for run_id dummy not found.",
        ):
            callback.on_retriever_end(
                [
                    Document(
                        page_content="document content 1",
                        metadata={"chunk_id": "1", "doc_uri": "uri1"},
                    )
                ],
                run_id="dummy",
            )
    finally:
        callback.flush()

def test_tool_success():
    callback = MlflowLangchainTracer()
    prompt = SystemMessagePromptTemplate.from_template("You are a nice assistant.") + "{question}"
    llm = ChatOpenAI()

    chain = prompt | llm | StrOutputParser()
    chain_tool = tool("chain_tool", chain)

    tool_input = {"question": "What up"}
    chain_tool.invoke(tool_input, config={"callbacks": [callback]})

    # str output is converted to _ChatResponse
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    spans = trace.data.spans
    assert len(spans) == 5

    # Tool
    tool_span = spans[0]
    assert tool_span.span_type == "TOOL"
    assert tool_span.inputs == tool_input
    assert tool_span.outputs is not None
    tool_span_id = tool_span.span_id

    # RunnableSequence
    runnable_sequence_span = spans[1]
    assert runnable_sequence_span.parent_id == tool_span_id
    assert runnable_sequence_span.span_type == "CHAIN"
    assert runnable_sequence_span.inputs == tool_input
    assert runnable_sequence_span.outputs is not None

    # PromptTemplate
    prompt_template_span = spans[2]
    assert prompt_template_span.span_type == "CHAIN"
    # LLM
    llm_span = spans[3]
    assert llm_span.span_type == "CHAT_MODEL"
    # StrOutputParser
    output_parser_span = spans[4]
    assert output_parser_span.span_type == "CHAIN"
    assert output_parser_span.outputs == [
        {"content": "You are a nice assistant.", "role": "system"},
        {"content": "What up", "role": "user"},
    ]

    _validate_trace_json_serialization(trace)

def test_tracer_thread_safe():
    tracer = MlflowLangchainTracer()

    def worker_function(worker_id):
        chain_run_id = str(uuid.uuid4())
        tracer.on_chain_start(
            {}, {"input": "test input"}, run_id=chain_run_id, name=f"chain_{worker_id}"
        )
        # wait for a random time (0.5 ~ 1s) to simulate real-world scenario
        time.sleep(random.random() / 2 + 0.5)
        tracer.on_chain_end({"output": "test output"}, run_id=chain_run_id)

    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="test-langchain-tracer") as executor:
        futures = [executor.submit(worker_function, i) for i in range(10)]
        for future in futures:
            future.result()

    traces = get_traces()
    assert len(traces) == 10
    assert all(len(trace.data.spans) == 1 for trace in traces)

def test_tracer_with_manual_traces():
    # Validate if the callback works properly when outer and inner spans
    # are created by fluent APIs.
    llm = ChatOpenAI()
    prompt = PromptTemplate(
        input_variables=["color"],
        template="What is the complementary color of {color}?",
    )

    # Inner spans are created within RunnableLambda
    def foo(s: str):
        with mlflow.start_span(name="foo_inner") as span:
            span.set_inputs(s)
            s = s.replace("red", "blue")
            s = bar(s)
            span.set_outputs(s)
        return s

    @mlflow.trace
    def bar(s):
        return s.replace("blue", "green")

    chain = RunnableLambda(foo) | prompt | llm | StrOutputParser()

    @mlflow.trace(name="parent", span_type="SPECIAL")
    def run(message):
        return chain.invoke(message, config={"callbacks": [MlflowLangchainTracer()]})

    response = run("red")
    expected_response = '[{"role": "user", "content": "What is the complementary color of green?"}]'
    assert response == expected_response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    spans = trace.data.spans
    assert spans[0].name == "parent"
    assert spans[1].name == "RunnableSequence"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "foo"
    assert spans[2].parent_id == spans[1].span_id
    assert spans[3].name == "foo_inner"
    assert spans[3].parent_id == spans[2].span_id
    assert spans[4].name == "bar"
    assert spans[4].parent_id == spans[3].span_id
    assert spans[5].name == "PromptTemplate"
    assert spans[5].parent_id == spans[1].span_id

async def test_tracer_with_manual_traces_async():
    llm = ChatOpenAI()
    prompt = PromptTemplate(
        input_variables=["color"],
        template="What is the complementary color of {color}?",
    )

    @mlflow.trace
    def manual_transform(s: str):
        return s.replace("red", "blue")

    chain = RunnableLambda(manual_transform) | prompt | llm | StrOutputParser()

    @mlflow.trace(name="parent")
    async def run(message):
        # run_inline=True ensures proper context propagation in async scenarios
        tracer = MlflowLangchainTracer(run_inline=True)
        return await chain.ainvoke(message, config={"callbacks": [tracer]})

    response = await run("red")
    expected_response = '[{"role": "user", "content": "What is the complementary color of blue?"}]'
    assert response == expected_response

    traces = get_traces()
    assert len(traces) == 1

    trace = traces[0]
    spans = trace.data.spans
    assert spans[0].name == "parent"
    assert spans[1].name == "RunnableSequence"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "manual_transform"
    assert spans[2].parent_id == spans[1].span_id

def test_chat_model_extracts_model_provider(_type, expected_provider):
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_chat_model_start(
        {},
        [[HumanMessage("test")]],
        run_id=run_id,
        name="test_chat_model",
        invocation_params={"model": "gpt-4", "_type": _type},
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "response"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    span = trace.data.spans[0]
    assert span.get_attribute(SpanAttributeKey.MODEL) == "gpt-4"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == expected_provider

def test_chat_model_no_provider_when_type_missing():
    callback = MlflowLangchainTracer()
    run_id = str(uuid.uuid4())
    callback.on_chat_model_start(
        {},
        [[HumanMessage("test")]],
        run_id=run_id,
        name="test_chat_model",
        invocation_params={"model": "gpt-4"},
    )
    callback.on_llm_end(
        LLMResult(generations=[[{"text": "response"}]]),
        run_id=run_id,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    span = trace.data.spans[0]
    assert span.get_attribute(SpanAttributeKey.MODEL) == "gpt-4"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) is None

def test_tracer_run_inline_parameter(run_tracer_inline):
    tracer = MlflowLangchainTracer(run_inline=run_tracer_inline)
    assert tracer.run_inline == run_tracer_inline


# --- tests/litellm/test_litellm_autolog.py ---

def test_litellm_tracing_success():
    mlflow.litellm.autolog()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
    )
    assert response.choices[0].message.content == '[{"role": "system", "content": "Hello"}]'

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-completion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].span_type == SpanType.LLM
    assert spans[0].inputs == {"messages": [{"role": "system", "content": "Hello"}]}
    assert spans[0].outputs == response.model_dump()
    assert spans[0].attributes["model"] == "gpt-4o-mini"
    assert spans[0].attributes["call_type"] == "completion"
    assert spans[0].attributes["cache_hit"] is None
    assert spans[0].attributes["response_cost"] > 0
    _assert_usage(
        spans[0].attributes,
        expected_prompt_tokens=9,
        expected_completion_tokens=12,
        expected_total_tokens=21,
    )

def test_litellm_tracing_failure():
    mlflow.litellm.autolog()

    with pytest.raises(litellm.exceptions.BadRequestError, match="LLM Provider"):
        litellm.completion(
            model="invalid-model",
            messages=[{"role": "system", "content": "Hello"}],
        )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "ERROR"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-completion"
    assert spans[0].status.status_code == "ERROR"
    assert spans[0].inputs == {"messages": [{"role": "system", "content": "Hello"}]}
    assert spans[0].outputs is None
    assert spans[0].attributes["model"] == "invalid-model"
    assert spans[0].attributes["response_cost"] == 0
    assert len(spans[0].events) == 1
    assert spans[0].events[0].name == "exception"

def test_litellm_tracing_streaming():
    mlflow.litellm.autolog()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
        stream=True,
    )

    chunks = [c.choices[0].delta.content for c in response]
    assert chunks == ["Hello", " world", None]

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-completion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].span_type == SpanType.LLM
    assert spans[0].inputs == {
        "messages": [{"role": "system", "content": "Hello"}],
        "stream": True,
    }
    assert spans[0].outputs["choices"][0]["message"]["content"] == "Hello world"
    assert spans[0].attributes["model"] == "gpt-4o-mini"
    _assert_usage(
        spans[0].attributes,
        expected_prompt_tokens=8,
        expected_completion_tokens=2,
        expected_total_tokens=10,
    )

async def test_litellm_tracing_async():
    mlflow.litellm.autolog()

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
    )
    assert response.choices[0].message.content == '[{"role": "system", "content": "Hello"}]'

    # Adding a sleep here to ensure that trace is logged.
    await asyncio.sleep(0.1)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-acompletion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].inputs == {"messages": [{"role": "system", "content": "Hello"}]}
    assert spans[0].outputs == response.model_dump()
    assert spans[0].attributes["model"] == "gpt-4o-mini"
    assert spans[0].attributes["call_type"] == "acompletion"
    assert spans[0].attributes["cache_hit"] is None
    assert spans[0].attributes["response_cost"] > 0
    _assert_usage(
        spans[0].attributes,
        expected_prompt_tokens=9,
        expected_completion_tokens=12,
        expected_total_tokens=21,
    )

async def test_litellm_tracing_async_streaming():
    mlflow.litellm.autolog()

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
        stream=True,
    )
    chunks: list[str | None] = []
    async for c in response:
        chunks.append(c.choices[0].delta.content)
        # Adding a sleep here to ensure that `content` in the span outputs is
        # consistently 'Hello World', not 'Hello' or ''.
        await asyncio.sleep(0.1)

    assert chunks == ["Hello", " world", None]

    # Await the logger task to ensure that the trace is logged.
    logger_task = next(
        task
        for task in asyncio.all_tasks()
        if "async_success_handler" in getattr(task.get_coro(), "__name__", "")
    )
    await logger_task

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-acompletion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].outputs["choices"][0]["message"]["content"] == "Hello world"

def test_litellm_tracing_with_parent_span():
    mlflow.litellm.autolog()

    with mlflow.start_span(name="parent"):
        litellm.completion(model="gpt-4o-mini", messages=[{"role": "system", "content": "Hello"}])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 2
    assert spans[0].name == "parent"
    assert spans[1].name == "litellm-completion"

def test_litellm_tracing_disable():
    mlflow.litellm.autolog()

    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    assert mlflow.get_trace(mlflow.get_last_active_trace_id()) is not None
    assert len(get_traces()) == 1

    mlflow.litellm.autolog(disable=True)
    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    # no additional trace should be created
    assert len(get_traces()) == 1

    mlflow.litellm.autolog(log_traces=False)
    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    # no additional trace should be created
    assert len(get_traces()) == 1


# --- tests/llama_index/test_llama_index_autolog.py ---

def test_autolog_should_not_generate_traces_during_logging_loading(single_index):
    mlflow.llama_index.autolog()

    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            name="model",
            pip_requirements=["mlflow"],
            engine_type="query",
        )
    loaded = mlflow.pyfunc.load_model(model_info.model_uri)

    assert len(get_traces()) == 0

    loaded.predict("Hello")
    assert len(get_traces()) == 1

def test_autolog_link_traces_to_loaded_model_engine(
    code_path, engine_type, engine_method, input_arg
):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    code_path,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type=engine_type,
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        getattr(model, engine_method)(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs[input_arg] == f"Hello {model_id}"

def test_autolog_link_traces_to_loaded_model_index_query(single_index, is_stream):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type="query",
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_query_engine(streaming=is_stream)
        response = engine.query(f"Hello {model_info.model_id}")
        if is_stream:
            response = "".join(response.response_gen)

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"

async def test_autolog_link_traces_to_loaded_model_index_query_async(single_index):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type="query",
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_query_engine()
        await engine.aquery(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"

def test_autolog_link_traces_to_loaded_model_index_chat(single_index, chat_mode):
    if llama_core_version >= Version("0.13.0") and chat_mode in [ChatMode.OPENAI, ChatMode.REACT]:
        pytest.skip("OpenAI and React chat modes are removed in 0.13.0")

    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index, name=f"model_{i}", pip_requirements=["mlflow"], engine_type="chat"
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_chat_engine(chat_mode=chat_mode)
        engine.chat(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["message"] == f"Hello {model_id}"

def test_autolog_link_traces_to_loaded_model_index_retriever(single_index):
    model_infos = []
    for i in range(3):
        with mlflow.start_run():
            model_infos.append(
                mlflow.llama_index.log_model(
                    single_index,
                    name=f"model_{i}",
                    pip_requirements=["mlflow"],
                    engine_type="retriever",
                )
            )

    mlflow.llama_index.autolog()
    for model_info in model_infos:
        model = mlflow.llama_index.load_model(model_info.model_uri)
        engine = model.as_retriever()
        engine.retrieve(f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 3
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert span.inputs["str_or_query_bundle"] == f"Hello {model_id}"

async def test_autolog_link_traces_to_loaded_model_workflow():
    mlflow.llama_index.autolog()
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            name="model",
            pip_requirements=["mlflow"],
        )
    loaded_workflow = mlflow.llama_index.load_model(model_info.model_uri)
    await loaded_workflow.run(topic=f"Hello {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id is not None
    # In llama-index >= 0.14.16, kwargs are flattened in span inputs
    if llama_core_version >= Version("0.14.16"):
        assert span.inputs["topic"] == f"Hello {model_id}"
    else:
        assert span.inputs["kwargs"]["topic"] == f"Hello {model_id}"

def test_autolog_link_traces_to_loaded_model_workflow_pyfunc():
    mlflow.llama_index.autolog()
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            name="model",
            pip_requirements=["mlflow"],
        )
    loaded_workflow = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_workflow.predict({"topic": f"Hello {model_info.model_id}"})

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id is not None
    if llama_core_version >= Version("0.14.16"):
        assert span.inputs["topic"] == f"Hello {model_id}"
    else:
        assert span.inputs["kwargs"]["topic"] == f"Hello {model_id}"

def test_autolog_link_traces_to_active_model():
    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(model_id=model.model_id)
    mlflow.llama_index.autolog()
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            "tests/llama_index/sample_code/simple_workflow.py",
            name="model",
            pip_requirements=["mlflow"],
        )
    loaded_workflow = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_workflow.predict({"topic": f"Hello {model_info.model_id}"})

    traces = get_traces()
    assert len(traces) == 1
    model_id = traces[0].info.request_metadata[TraceMetadataKey.MODEL_ID]
    assert model_id == model.model_id
    assert model_id != model_info.model_id


# --- tests/llama_index/test_llama_index_model_export.py ---

def test_llama_index_load_with_model_config(single_index):
    from llama_index.core.response_synthesizers import Refine

    with mlflow.start_run():
        logged_model = mlflow.llama_index.log_model(
            single_index,
            name="model",
            engine_type="query",
            model_config={"response_mode": "refine"},
        )

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    engine = loaded_model.get_raw_model()

    assert isinstance(engine._response_synthesizer, Refine)

def test_format_predict_input_correct(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    assert isinstance(
        wrapped_model._format_predict_input(pd.DataFrame({"query_str": ["hi"]})), QueryBundle
    )
    assert isinstance(wrapped_model._format_predict_input(np.array(["hi"])), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": ["hi"]}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": "hi"}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input(["hi"]), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input("hi"), QueryBundle)

def test_format_predict_input_incorrect_schema(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    exception_error = (
        r"__init__\(\) got an unexpected keyword argument 'incorrect'"
        if Version(llama_index.core.__version__) >= Version("0.11.0")
        else r"missing 1 required positional argument"
    )

    with pytest.raises(TypeError, match=exception_error):
        wrapped_model._format_predict_input(pd.DataFrame({"incorrect": ["hi"]}))
    with pytest.raises(TypeError, match=exception_error):
        wrapped_model._format_predict_input({"incorrect": ["hi"]})

def test_format_predict_input_correct_schema_complex(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    payload = {
        "query_str": "hi",
        "image_path": "some/path",
        "custom_embedding_strs": [["a"]],
        "embedding": [[1.0]],
    }
    assert isinstance(wrapped_model._format_predict_input(pd.DataFrame(payload)), QueryBundle)
    payload.update({
        "custom_embedding_strs": ["a"],
        "embedding": [1.0],
    })
    assert isinstance(wrapped_model._format_predict_input(payload), QueryBundle)

def test_query_engine_predict(single_index, with_input_example, payload):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            name="model",
            input_example=payload if with_input_example else None,
            engine_type="query",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])

    model = mlflow.pyfunc.load_model(model_info.model_uri)

    prediction = model.predict(payload)
    assert isinstance(prediction, str)
    assert prediction.startswith('[{"role": "system",')

def test_query_engine_predict_list(single_index, with_input_example, payload):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            name="model",
            input_example=payload if with_input_example else None,
            engine_type="query",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = model.predict(payload)

    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert all(isinstance(p, str) for p in predictions)
    assert all(p.startswith('[{"role": "system",') for p in predictions)

def test_query_engine_predict_numeric(model_path, single_index, with_input_example):
    payload = 1

    input_example = payload if with_input_example else None
    if with_input_example:
        with pytest.raises(ValueError, match="Unsupported input type"):
            mlflow.llama_index.save_model(
                llama_index_model=single_index,
                input_example=input_example,
                path=model_path,
                engine_type="query",
            )
    else:
        mlflow.llama_index.save_model(
            llama_index_model=single_index, path=model_path, engine_type="query"
        )
        model = mlflow.pyfunc.load_model(model_path)
        with pytest.raises(ValueError, match="Unsupported input type"):
            _ = model.predict(payload)

def test_chat_engine_predict(single_index, with_input_example, payload):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            name="model",
            input_example=payload if with_input_example else None,
            engine_type="chat",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = model.predict(payload)
    assert isinstance(prediction, str)
    # a default prompt is added in llama-index 0.13.0
    # https://github.com/run-llama/llama_index/blob/1e02c7a2324838f7bd5a52c811d35c30dc6a6bd2/llama-index-core/llama_index/core/chat_engine/condense_plus_context.py#L40
    assert '{"role": "user", "content": "string"}' in prediction

def test_chat_engine_dict_raises(model_path, single_index, with_input_example):
    payload = {
        "message": "string",
        "key_that_no_exist": [str(ChatMessage(role="user", content="string"))],
    }

    input_example = payload if with_input_example else None
    if with_input_example:
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            mlflow.llama_index.save_model(
                llama_index_model=single_index,
                input_example=input_example,
                path=model_path,
                engine_type="chat",
            )
    else:
        mlflow.llama_index.save_model(
            llama_index_model=single_index,
            input_example=input_example,
            path=model_path,
            engine_type="chat",
        )

        model = mlflow.pyfunc.load_model(model_path)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _ = model.predict(payload)

def test_retriever_engine_predict(single_index, with_input_example):
    payload = "string"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            single_index,
            name="model",
            input_example=payload if with_input_example else None,
            engine_type="retriever",
        )

    if with_input_example:
        assert model_info.signature.inputs is not None
        # TODO: Inferring signature from retriever output fails because the schema
        # does not allow None value. This is a bug in the schema inference.
        # assert model_info.signature.outputs is not None

    model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = model.predict(payload)
    assert all(p["class_name"] == "NodeWithScore" for p in predictions)

def test_save_load_index_as_code_index(index_code_path, vector_store_class):
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            engine_type="query",
            input_example="hi",
        )

    artifact_path = Path(_download_artifact_from_uri(model_info.model_uri))
    assert os.path.exists(artifact_path / os.path.basename(index_code_path))
    assert not os.path.exists(artifact_path / "index")
    assert os.path.exists(artifact_path / "settings.json")

    loaded_index = mlflow.llama_index.load_model(model_info.model_uri)
    assert isinstance(loaded_index.vector_store, vector_store_class)

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), BaseQueryEngine)
    assert _TEST_QUERY in pyfunc_loaded_model.predict(_TEST_QUERY)

def test_save_load_query_engine_as_code():
    index_code_path = "tests/llama_index/sample_code/query_engine_with_reranker.py"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            input_example="hi",
        )

    loaded_engine = mlflow.llama_index.load_model(model_info.model_uri)
    assert isinstance(loaded_engine, BaseQueryEngine)
    processors = loaded_engine._node_postprocessors
    assert len(processors) == 2
    assert processors[0].__class__.__name__ == "LLMRerank"
    assert processors[1].__class__.__name__ == "CustomNodePostprocessor"

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model._model_impl, QueryEngineWrapper)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), BaseQueryEngine)
    assert pyfunc_loaded_model.predict(_TEST_QUERY) != ""
    custom_processor = pyfunc_loaded_model.get_raw_model()._node_postprocessors[1]
    assert custom_processor.call_count == 1

def test_save_load_chat_engine_as_code():
    index_code_path = "tests/llama_index/sample_code/basic_chat_engine.py"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            input_example="hi",
        )

    loaded_engine = mlflow.llama_index.load_model(model_info.model_uri)
    # The sample code sets chat mode to SIMPLE, so it should be a SimpleChatEngine
    assert isinstance(loaded_engine, SimpleChatEngine)

    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model._model_impl, ChatEngineWrapper)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), SimpleChatEngine)
    assert pyfunc_loaded_model.predict(_TEST_QUERY) != ""

def test_save_load_as_code_with_model_config(index_code_path, model_config):
    with mlflow.start_run():
        logged_model = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            model_config=model_config,
        )

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    engine = loaded_model.get_raw_model()
    assert engine._llm.model == "gpt-4o-mini"
    assert engine._llm.temperature == 0.7

async def test_save_load_workflow_as_code():
    from llama_index.core.workflow import Workflow

    index_code_path = "tests/llama_index/sample_code/simple_workflow.py"
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index_code_path,
            name="model",
            input_example={"topic": "pirates"},
        )

    # Signature
    assert model_info.signature.inputs == Schema([ColSpec(type=DataType.string, name="topic")])
    assert model_info.signature.outputs == Schema([ColSpec(DataType.string)])

    # Native inference
    loaded_workflow = mlflow.llama_index.load_model(model_info.model_uri)
    assert isinstance(loaded_workflow, Workflow)
    result = await loaded_workflow.run(topic="pirates")
    assert isinstance(result, str)
    assert "pirates" in result

    # Pyfunc inference
    pyfunc_loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert isinstance(pyfunc_loaded_model.get_raw_model(), Workflow)
    result = pyfunc_loaded_model.predict({"topic": "pirates"})
    assert isinstance(result, str)
    assert "pirates" in result

    # Batch inference
    batch_result = pyfunc_loaded_model.predict([
        {"topic": "pirates"},
        {"topic": "ninjas"},
        {"topic": "robots"},
    ])
    assert len(batch_result) == 3
    assert all(isinstance(r, str) for r in batch_result)

    # Serve
    inference_payload = load_serving_example(model_info.model_uri)

    with pyfunc_scoring_endpoint(
        model_uri=model_info.model_uri,
        extra_args=["--env-manager", "local"],
    ) as endpoint:
        # Single input
        response = endpoint.invoke(inference_payload, content_type=CONTENT_TYPE_JSON)
        assert response.status_code == 200, response.text
        assert response.json()["predictions"] == result

        # Batch input
        df = pd.DataFrame({"topic": ["pirates", "ninjas", "robots"]})
        response = endpoint.invoke(
            json.dumps({"dataframe_split": df.to_dict(orient="split")}),
            content_type=CONTENT_TYPE_JSON,
        )
        assert response.status_code == 200, response.text
        assert response.json()["predictions"] == batch_result


# --- tests/llama_index/test_llama_index_pyfunc_wrapper.py ---

def test_format_predict_input_str_chat(single_index):
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input("string")
    assert formatted_data == "string"

def test_format_predict_input_dict_chat(single_index):
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input({"query": "string"})
    assert isinstance(formatted_data, dict)

def test_format_predict_input_message_history_chat(single_index):
    payload = {
        "message": "string",
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [{"role": "user", "content": "hi"}] * 3,
    }
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(payload)

    assert isinstance(formatted_data, dict)
    assert formatted_data["message"] == payload["message"]
    assert isinstance(formatted_data[_CHAT_MESSAGE_HISTORY_PARAMETER_NAME], list)
    assert all(
        isinstance(x, ChatMessage) for x in formatted_data[_CHAT_MESSAGE_HISTORY_PARAMETER_NAME]
    )

def test_format_predict_input_message_history_chat_iterable(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)

    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")

    assert isinstance(formatted_data, list)
    assert formatted_data[0]["query"] == data[0]["query"]
    assert isinstance(formatted_data[0][_CHAT_MESSAGE_HISTORY_PARAMETER_NAME], list)
    assert all(
        isinstance(x, ChatMessage) for x in formatted_data[0][_CHAT_MESSAGE_HISTORY_PARAMETER_NAME]
    )

def test_format_predict_input_message_history_chat_invalid_type(single_index):
    payload = {
        "message": "string",
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: ["invalid history string", "user: hi"],
    }
    wrapped_model = create_pyfunc_wrapper(single_index, CHAT_ENGINE_NAME)
    with pytest.raises(ValueError, match="It must be a list of dicts"):
        _ = wrapped_model._format_predict_input(payload)

def test_format_predict_input_no_iterable_query(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, QUERY_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, QueryBundle)

def test_format_predict_input_iterable_query(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, QUERY_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)

    assert isinstance(formatted_data, list)
    assert all(isinstance(x, QueryBundle) for x in formatted_data)

def test_format_predict_input_no_iterable_retriever(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, RETRIEVER_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, QueryBundle)

def test_format_predict_input_iterable_retriever(single_index, data):
    wrapped_model = create_pyfunc_wrapper(single_index, RETRIEVER_ENGINE_NAME)
    formatted_data = wrapped_model._format_predict_input(data)
    assert isinstance(formatted_data, list)
    assert all(isinstance(x, QueryBundle) for x in formatted_data)

def test_format_predict_input_correct(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    assert isinstance(
        wrapped_model._format_predict_input(pd.DataFrame({"query_str": ["hi"]})), QueryBundle
    )
    assert isinstance(wrapped_model._format_predict_input(np.array(["hi"])), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": ["hi"]}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input({"query_str": "hi"}), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input(["hi"]), QueryBundle)
    assert isinstance(wrapped_model._format_predict_input("hi"), QueryBundle)

def test_format_predict_input_correct_schema_complex(single_index, engine_type):
    wrapped_model = create_pyfunc_wrapper(single_index, engine_type)

    payload = {
        "query_str": "hi",
        "image_path": "some/path",
        "custom_embedding_strs": [["a"]],
        "embedding": [[1.0]],
    }
    assert isinstance(wrapped_model._format_predict_input(pd.DataFrame(payload)), QueryBundle)
    payload.update({
        "custom_embedding_strs": ["a"],
        "embedding": [1.0],
    })
    assert isinstance(wrapped_model._format_predict_input(payload), QueryBundle)

def test_spark_udf_retriever_and_query_engine(model_path, spark, single_index, engine_type, input):
    mlflow.llama_index.save_model(
        llama_index_model=single_index,
        engine_type=engine_type,
        path=model_path,
        input_example=input,
    )
    udf = mlflow.pyfunc.spark_udf(spark, model_path, result_type="string")
    df = spark.createDataFrame([{"query_str": "hi"}])
    df = df.withColumn("predictions", udf())
    pdf = df.toPandas()
    assert len(pdf["predictions"].tolist()) == 1
    assert isinstance(pdf["predictions"].tolist()[0], str)

def test_spark_udf_chat(model_path, spark, single_index):
    engine_type = "chat"
    input = pd.DataFrame({
        "message": ["string"],
        _CHAT_MESSAGE_HISTORY_PARAMETER_NAME: [[{"role": "user", "content": "string"}]],
    })
    mlflow.llama_index.save_model(
        llama_index_model=single_index,
        engine_type=engine_type,
        path=model_path,
        input_example=input,
    )
    udf = mlflow.pyfunc.spark_udf(spark, model_path, result_type="string")
    df = spark.createDataFrame(input)
    df = df.withColumn("predictions", udf())
    pdf = df.toPandas()
    assert len(pdf["predictions"].tolist()) == 1
    assert isinstance(pdf["predictions"].tolist()[0], str)

async def test_wrap_workflow():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result=f"Hi, {ev.name}!")

    w = MyWorkflow(timeout=10, verbose=False)
    wrapper = create_pyfunc_wrapper(w)
    assert wrapper.get_raw_model() == w

    result = wrapper.predict({"name": "Alice"})
    assert result == "Hi, Alice!"

    results = wrapper.predict([
        {"name": "Bob"},
        {"name": "Charlie"},
    ])
    assert results == ["Hi, Bob!", "Hi, Charlie!"]

    results = wrapper.predict(pd.DataFrame({"name": ["David"]}))
    assert results == "Hi, David!"

    results = wrapper.predict(pd.DataFrame({"name": ["Eve", "Frank"]}))
    assert results == ["Hi, Eve!", "Hi, Frank!"]

async def test_wrap_workflow_raise_exception():
    from llama_index.core.workflow import (
        StartEvent,
        StopEvent,
        Workflow,
        WorkflowRuntimeError,
        step,
    )

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            raise ValueError("Expected error")

    w = MyWorkflow(timeout=10, verbose=False)
    wrapper = create_pyfunc_wrapper(w)

    with pytest.raises(
        (
            ValueError,  # llama_index < 0.12.1
            WorkflowRuntimeError,  # llama_index >= 0.12.1
        ),
        match="Expected error",
    ):
        wrapper.predict({"name": "Alice"})


# --- tests/llama_index/test_llama_index_tracer.py ---

def test_trace_llm_complete_stream():
    model_name = "gpt-3.5-turbo"
    llm = OpenAI(model=model_name)

    response_gen = llm.stream_complete("Hello", stream_options={"include_usage": True})
    # No trace should be created until the generator is consumed
    assert len(get_traces()) == 0
    assert inspect.isgenerator(response_gen)

    response = [r.text for r in response_gen]
    assert response == ["Hello", "Hello world"]

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.stream_complete"
    assert spans[0].span_type == SpanType.LLM
    assert spans[0].inputs == {
        "args": ["Hello"],
        "kwargs": {"stream_options": {"include_usage": True}},
    }
    assert spans[0].outputs["text"] == "Hello world"

    attr = spans[0].attributes
    assert (
        attr["usage"].items()
        >= {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }.items()
    )
    assert attr[SpanAttributeKey.CHAT_USAGE] == {
        TokenUsageKey.INPUT_TOKENS: 9,
        TokenUsageKey.OUTPUT_TOKENS: 12,
        TokenUsageKey.TOTAL_TOKENS: 21,
    }
    assert attr["prompt"] == "Hello"
    assert attr["invocation_params"]["model_name"] == model_name
    assert attr["model_dict"]["model"] == model_name
    assert spans[0].model_name == model_name
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 9,
        TokenUsageKey.OUTPUT_TOKENS: 12,
        TokenUsageKey.TOTAL_TOKENS: 21,
    }

def test_trace_llm_chat_stream():
    llm = OpenAI()
    message = ChatMessage(role="system", content="Hello")

    response_gen = llm.stream_chat([message], stream_options={"include_usage": True})
    # No trace should be created until the generator is consumed
    assert len(get_traces()) == 0
    assert inspect.isgenerator(response_gen)

    chunks = list(response_gen)
    assert len(chunks) == 2
    assert all(isinstance(c.message, ChatMessage) for c in chunks)
    assert [c.message.content for c in chunks] == ["Hello", "Hello world"]

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "OpenAI.stream_chat"
    assert spans[0].span_type == SpanType.CHAT_MODEL
    assert spans[0].model_name == llm.metadata.model_name

    content_json = _get_llm_input_content_json("Hello")
    assert spans[0].inputs == {
        "messages": [{"role": "system", **content_json, "additional_kwargs": {}}],
        "kwargs": {"stream_options": {"include_usage": True}},
    }
    # `additional_kwargs` was broken until 0.1.30 release of llama-index-llms-openai
    expected_kwargs = (
        {"completion_tokens": 12, "prompt_tokens": 9, "total_tokens": 21}
        if llama_oai_version >= Version("0.1.30")
        else {}
    )
    output_content_json = _get_llm_input_content_json("Hello world")
    assert spans[0].outputs == {
        "message": {
            "role": "assistant",
            **output_content_json,
            "additional_kwargs": {},
        },
        "raw": ANY,
        "delta": " world",
        "logprobs": None,
        "additional_kwargs": expected_kwargs,
    }

    attr = spans[0].attributes
    assert (
        attr["usage"].items()
        >= {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }.items()
    )
    assert attr[SpanAttributeKey.CHAT_USAGE] == {
        TokenUsageKey.INPUT_TOKENS: 9,
        TokenUsageKey.OUTPUT_TOKENS: 12,
        TokenUsageKey.TOTAL_TOKENS: 21,
    }
    assert attr["invocation_params"]["model_name"] == llm.metadata.model_name
    assert attr["model_dict"]["model"] == llm.metadata.model_name
    assert spans[0].model_name == llm.metadata.model_name
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 9,
        TokenUsageKey.OUTPUT_TOKENS: 12,
        TokenUsageKey.TOTAL_TOKENS: 21,
    }

def test_trace_retriever(multi_index, is_async):
    retriever = VectorIndexRetriever(multi_index, similarity_top_k=3)

    if is_async:
        retrieved = asyncio.run(retriever.aretrieve("apple"))
    else:
        retrieved = retriever.retrieve("apple")
    assert len(retrieved) == 1

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert len(spans) == 4
    for i in range(1, 4):
        assert spans[i].parent_id == spans[i - 1].span_id

    assert spans[0].name.endswith("Retriever.aretrieve" if is_async else "Retriever.retrieve")
    assert spans[0].span_type == SpanType.RETRIEVER
    assert spans[0].inputs == {"str_or_query_bundle": "apple"}
    assert len(spans[0].outputs) == 1

    if Version(llama_index.core.__version__) >= Version("0.12.5"):
        retrieved_text = retrieved[0].node.text
    else:
        retrieved_text = retrieved[0].text
    assert spans[0].outputs[0]["page_content"] == retrieved_text

    assert spans[1].name.startswith("VectorIndexRetriever")
    assert spans[1].span_type == SpanType.RETRIEVER
    assert spans[1].inputs["query_bundle"]["query_str"] == "apple"
    assert spans[1].outputs == spans[0].outputs

    assert "Embedding" in spans[2].name
    assert spans[2].span_type == SpanType.EMBEDDING
    assert spans[2].inputs == {"query": "apple"}
    assert len(spans[2].outputs) == 1536  # embedding size
    assert spans[2].attributes["model_name"] == Settings.embed_model.model_name
    assert spans[2].model_name == Settings.embed_model.model_name

    assert "Embedding" in spans[3].name
    assert spans[3].span_type == SpanType.EMBEDDING
    assert spans[3].inputs == {"query": "apple"}
    assert len(spans[3].outputs) == 1536  # embedding size
    assert spans[3].attributes["model_name"] == Settings.embed_model.model_name
    assert spans[3].model_name == Settings.embed_model.model_name

def test_trace_query_engine(multi_index, is_stream, is_async):
    if is_stream and is_async:
        pytest.skip("Async stream is not supported yet")

    engine = multi_index.as_query_engine(streaming=is_stream)

    if is_stream:
        response = engine.query("Hello")
        assert isinstance(response, StreamingResponse)
        response = "".join(response.response_gen)
        assert response == "Hello world"
    else:
        response = asyncio.run(engine.aquery("Hello")) if is_async else engine.query("Hello")
        assert response.response.startswith('[{"role": "system", "content": "You are an')
        response = asdict(response)
        if Version(llama_index.core.__version__) > Version("0.10.68"):
            response["source_nodes"] = [n.dict() for n in response["source_nodes"]]

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    # Async methods have "a" prefix
    prefix = "a" if is_async else ""

    # Validate span attributes for some key spans
    spans = traces[0].data.spans
    assert spans[0].name.endswith(f"QueryEngine.{prefix}query")
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].inputs == {"str_or_query_bundle": "Hello"}
    assert spans[0].outputs == response

def test_trace_chat_engine(multi_index, is_stream, is_async):
    if is_stream:
        if is_async:
            pytest.skip("Async stream is not supported yet")

        # Skip streaming test for llama-index <0.13 due to race condition with OpenAIAgent
        # where child spans are created after root span completes, causing incomplete traces
        if llama_core_version < Version("0.13.0"):
            pytest.skip("Streaming chat engine test is flaky for llama-index <0.13")

    engine = multi_index.as_chat_engine()

    if is_stream:
        response_gen = engine.stream_chat("Hello").response_gen
        response = "".join(response_gen)
        assert response == "Hello world"
    else:
        response = asyncio.run(engine.achat("Hello")) if is_async else engine.chat("Hello")
        # a default prompt is added in llama-index 0.13.0
        # https://github.com/run-llama/llama_index/blob/1e02c7a2324838f7bd5a52c811d35c30dc6a6bd2/llama-index-core/llama_index/core/chat_engine/condense_plus_context.py#L40
        assert '{"role": "user", "content": "Hello"}' in response.response

    # Since chat engine is a complex agent-based system, it is challenging to strictly
    # validate the trace structure and attributes. The detailed validation is done in
    # other tests for individual components.
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK
    root_span = traces[0].data.spans[0]
    assert root_span.inputs == {"message": "Hello"}

def test_tracer_handle_tracking_uri_update(tmp_path):
    OpenAI().complete("Hello")
    assert len(get_traces()) == 1

    # Set different tracking URI and initialize the tracer
    with _use_tracking_uri(tmp_path / "dummy"):
        assert len(get_traces()) == 0

        # The new trace will be logged to the updated tracking URI
        OpenAI().complete("Hello")
        assert len(get_traces()) == 1

async def test_tracer_simple_workflow():
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="Hi, world!")

    w = MyWorkflow(timeout=10, verbose=False)
    await w.run()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK
    assert all(s.status.status_code == SpanStatusCode.OK for s in traces[0].data.spans)

async def test_tracer_parallel_workflow():
    from llama_index.core.workflow import (
        Context,
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )

    class ProcessEvent(Event):
        data: str

    class ResultEvent(Event):
        result: str

    class ParallelWorkflow(Workflow):
        @step
        async def start(self, ctx: Context, ev: StartEvent) -> ProcessEvent:
            await context_set(ctx, "num_to_collect", len(ev.inputs))
            for item in ev.inputs:
                ctx.send_event(ProcessEvent(data=item))
            return None

        @step(num_workers=3)
        async def process_data(self, ev: ProcessEvent) -> ResultEvent:
            # Simulate some time-consuming processing
            await asyncio.sleep(random.randint(1, 2))
            return ResultEvent(result=ev.data)

        @step
        async def combine_results(self, ctx: Context, ev: ResultEvent) -> StopEvent:
            num_to_collect = await context_get(ctx, "num_to_collect")
            results = ctx.collect_events(ev, [ResultEvent] * num_to_collect)
            if results is None:
                return None

            combined_result = ", ".join(sorted([event.result for event in results]))
            return StopEvent(result=combined_result)

    w = ParallelWorkflow()
    result = await w.run(inputs=["apple", "grape", "orange", "banana"])
    assert result == "apple, banana, grape, orange"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK
    for s in traces[0].data.spans:
        assert s.status.status_code == SpanStatusCode.OK

    root_span = traces[0].data.spans[0]
    # In llama-index >= 0.14.16, kwargs are flattened in span inputs
    if llama_core_version >= Version("0.14.16"):
        expected_inputs = {"inputs": ["apple", "grape", "orange", "banana"]}
    else:
        expected_inputs = {"kwargs": {"inputs": ["apple", "grape", "orange", "banana"]}}
    # assert that the inputs are a superset of the expected inputs.
    # this is to make the test resilient to framework changes which may add additional inputs.
    assert all(root_span.inputs.get(k) == v for k, v in expected_inputs.items())
    # in llama-index < 0.14, outputs are a string
    if isinstance(root_span.outputs, str):
        assert root_span.outputs == "apple, banana, grape, orange"
    else:
        assert root_span.outputs["result"] == "apple, banana, grape, orange"

async def test_tracer_parallel_workflow_with_custom_spans():
    from llama_index.core.workflow import (
        Context,
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )

    class ProcessEvent(Event):
        data: str

    class ResultEvent(Event):
        result: str

    class ParallelWorkflow(Workflow):
        @step
        async def start(self, ctx: Context, ev: StartEvent) -> ProcessEvent:
            await context_set(ctx, "num_to_collect", len(ev.inputs))
            for item in ev.inputs:
                ctx.send_event(ProcessEvent(data=item))
            return None

        @step(num_workers=3)
        async def process_data(self, ev: ProcessEvent) -> ResultEvent:
            # Simulate some time-consuming processing
            await asyncio.sleep(random.randint(1, 2))
            with mlflow.start_span(name="custom_inner_span_worker"):
                pass
            return ResultEvent(result=ev.data)

        @step
        async def combine_results(self, ctx: Context, ev: ResultEvent) -> StopEvent:
            num_to_collect = await context_get(ctx, "num_to_collect")
            results = ctx.collect_events(ev, [ResultEvent] * num_to_collect)
            if results is None:
                return None

            with mlflow.start_span(name="custom_inner_result_span") as span:
                span.set_inputs(results)
                combined_result = ", ".join(sorted([event.result for event in results]))
                span.set_outputs(combined_result)
            return StopEvent(result=combined_result)

    w = ParallelWorkflow()
    inputs = ["apple", "grape", "orange", "banana"]

    result = await w.run(inputs=inputs)
    assert result == "apple, banana, grape, orange"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK

    spans = traces[0].data.spans
    assert all(s.status.status_code == SpanStatusCode.OK for s in spans)

    workflow_span = spans[0]
    if llama_core_version >= Version("0.14.16"):
        expected_inputs = {"inputs": inputs}
    else:
        expected_inputs = {"kwargs": {"inputs": inputs}}
    assert all(workflow_span.inputs.get(k) == v for k, v in expected_inputs.items())
    if isinstance(workflow_span.outputs, str):
        assert workflow_span.outputs == result
    else:
        assert workflow_span.outputs["result"] == result

    inner_worker_spans = [s for s in spans if s.name.startswith("custom_inner_span_worker")]
    assert len(inner_worker_spans) == len(inputs)

    inner_result_span = next(s for s in spans if s.name == "custom_inner_result_span")
    assert inner_result_span.inputs is not None
    assert inner_result_span.outputs == result

async def test_stream_resolver_with_async_generator(should_close):
    async def async_generator():
        yield "chunk1"
        yield "chunk2"

    resolver = StreamResolver()
    tracer = _get_tracer(__name__)

    agen = async_generator()
    if should_close:
        async for _ in agen:
            pass

    with tracer.start_as_current_span("test_closed_async") as otel_span:
        from mlflow.entities.span import LiveSpan

        trace_id = f"{otel_span.context.trace_id:032x}"
        span = LiveSpan(otel_span=otel_span, trace_id=trace_id)

        # Should detect that the generator is closed and return False
        result = resolver.register_stream_span(span, agen)
        assert result == (not should_close)


# --- tests/openai/test_genai_semconv_converter.py ---

def test_autolog_basic(client, capture_otel_export, api):
    exporter, processor = capture_otel_export

    mlflow.openai.autolog()
    if api == "chat_completions":
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model=MODEL,
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            stop=["\n", "END"],
        )
    else:
        client.responses.create(input="Hi", model=MODEL, temperature=0.5)

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == MODEL
    assert chat_span.attributes["gen_ai.request.temperature"] == 0.5

    if api == "chat_completions":
        assert chat_span.attributes["gen_ai.request.top_p"] == 0.9
        assert chat_span.attributes["gen_ai.request.max_tokens"] == 100
        assert list(chat_span.attributes["gen_ai.request.stop_sequences"]) == ["\n", "END"]

    input_msgs = json.loads(chat_span.attributes["gen_ai.input.messages"])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"
    assert input_msgs[0]["parts"][0]["content"] == "Hi"

    output_msgs = json.loads(chat_span.attributes["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"

    assert chat_span.attributes["gen_ai.response.model"] == MODEL
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)

def test_autolog_streaming(client, capture_otel_export, api):
    exporter, processor = capture_otel_export

    mlflow.openai.autolog()
    if api == "chat_completions":
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model=MODEL,
            stream=True,
        )
        for _ in stream:
            pass
    else:
        stream = client.responses.create(input="Hi", model=MODEL, stream=True)
        for _ in stream:
            pass

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == MODEL

    input_msgs = json.loads(chat_span.attributes["gen_ai.input.messages"])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"
    assert input_msgs[0]["parts"][0]["content"] == "Hi"

    output_msgs = json.loads(chat_span.attributes["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"

    assert chat_span.attributes["gen_ai.response.model"] == MODEL
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)


# --- tests/openai/test_openai_agent_autolog.py ---

async def test_autolog_agent_tool_exception():
    mlflow.openai.autolog()

    DUMMY_RESPONSES = [
        Response(
            id="123",
            created_at=12345678.0,
            error=None,
            model="gpt-4o-mini",
            object="response",
            instructions="You are an agent.",
            output=[
                ResponseFunctionToolCall(
                    id="123",
                    arguments="{}",
                    call_id="123",
                    name="always_fail",
                    type="function_call",
                    status="completed",
                )
            ],
            tools=[
                FunctionTool(
                    name="always_fail",
                    parameters={"type": "object", "properties": {}, "required": []},
                    type="function",
                    strict=False,
                ),
            ],
            tool_choice="auto",
            temperature=1,
            parallel_tool_calls=True,
        ),
    ]

    @function_tool(failure_error_function=None)  # Set error function None to avoid retry
    def always_fail():
        raise Exception("This function always fails")

    set_dummy_client(DUMMY_RESPONSES * 3)

    agent = Agent(name="Agent", instructions="You are an agent", tools=[always_fail])

    with pytest.raises(Exception, match="This function always fails"):
        await Runner.run(agent, [{"role": "user", "content": "Hi!"}])

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    spans = trace.data.spans
    assert len(spans) == 4  # 1 root + 1 function call + 1 get_chat_completion + 1 Completions
    assert spans[3].span_type == SpanType.TOOL
    assert spans[3].status.status_code == "ERROR"
    assert spans[3].status.description == "Error running tool"
    assert spans[3].events[0].name == "exception"


# --- tests/openai/test_openai_autolog.py ---

async def test_chat_completions_autolog_under_current_active_span(client):
    # If a user have an active span, the autologging should create a child span under it.
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_span(name="parent"):
        for _ in range(3):
            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0,
            )

            if client._is_async:
                await response

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 4
    parent_span = trace.data.spans[0]
    assert parent_span.name == "parent"
    child_span = trace.data.spans[1]
    assert child_span.name == "AsyncCompletions" if client._is_async else "Completions"
    assert child_span.inputs == {"messages": messages, "model": "gpt-4o-mini", "temperature": 0}
    assert child_span.outputs["id"] == "chatcmpl-123"
    assert child_span.parent_id == parent_span.span_id

    # Token usage should be aggregated correctly
    assert trace.info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 27,
        TokenUsageKey.OUTPUT_TOKENS: 36,
        TokenUsageKey.TOTAL_TOKENS: 63,
    }

async def test_chat_completions_autolog_streaming(client, include_usage):
    mlflow.openai.autolog()

    stream_options_supported = Version(openai.__version__) >= Version("1.26")

    if not stream_options_supported and include_usage:
        pytest.skip("OpenAI SDK version does not support usage tracking in streaming")

    messages = [{"role": "user", "content": "test"}]

    input_params = {
        "messages": messages,
        "model": "gpt-4o-mini",
        "temperature": 0,
        "stream": True,
    }
    if stream_options_supported:
        input_params["stream_options"] = {"include_usage": include_usage}

    stream = client.chat.completions.create(**input_params)

    if client._is_async:
        async for _ in await stream:
            pass
    else:
        for _ in stream:
            pass

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == input_params

    # Reconstructed response from streaming chunks
    assert isinstance(span.outputs, dict)
    assert span.outputs["id"] == "chatcmpl-123"
    assert span.outputs["object"] == "chat.completion"
    assert span.outputs["model"] == "gpt-4o-mini"
    assert span.outputs["system_fingerprint"] == "fp_44709d6fcb"
    assert "choices" in span.outputs
    assert span.outputs["choices"][0]["message"]["role"] == "assistant"
    assert span.outputs["choices"][0]["message"]["content"] == "Hello world"

    # Usage should be preserved when include_usage=True
    if include_usage:
        assert "usage" in span.outputs
        assert span.outputs["usage"]["prompt_tokens"] == 9
        assert span.outputs["usage"]["completion_tokens"] == 12
        assert span.outputs["usage"]["total_tokens"] == 21

    stream_event_data = trace.data.spans[0].events
    assert stream_event_data[0].name == "mlflow.chunk.item.0"
    chunk_1 = json.loads(stream_event_data[0].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_1["id"] == "chatcmpl-123"
    assert chunk_1["choices"][0]["delta"]["content"] == "Hello"
    assert stream_event_data[1].name == "mlflow.chunk.item.1"
    chunk_2 = json.loads(stream_event_data[1].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_2["id"] == "chatcmpl-123"
    assert chunk_2["choices"][0]["delta"]["content"] == " world"

    if include_usage:
        assert trace.info.token_usage == {
            TokenUsageKey.INPUT_TOKENS: 9,
            TokenUsageKey.OUTPUT_TOKENS: 12,
            TokenUsageKey.TOTAL_TOKENS: 21,
        }

async def test_chat_completions_autolog_tracing_error(client):
    mlflow.openai.autolog()
    messages = [{"role": "user", "content": "test"}]
    with pytest.raises(openai.UnprocessableEntityError, match="Input should be less"):  # noqa: PT012
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=5.0,
        )

        if client._is_async:
            await response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "ERROR"

    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.name == "AsyncCompletions" if client._is_async else "Completions"
    assert span.inputs["messages"][0]["content"] == "test"
    assert span.outputs is None

    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.type"] == "UnprocessableEntityError"

async def test_chat_completions_autolog_tracing_error_with_parent_span(client):
    mlflow.openai.autolog()

    if client._is_async:

        @mlflow.trace
        async def create_completions(text: str) -> str:
            try:
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model="gpt-4o-mini",
                    temperature=5.0,
                )
                return response.choices[0].delta.content
            except openai.OpenAIError as e:
                raise MlflowException("Failed to create completions") from e

        with pytest.raises(MlflowException, match="Failed to create completions"):
            await create_completions("test")

    else:

        @mlflow.trace
        def create_completions(text: str) -> str:
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model="gpt-4o-mini",
                    temperature=5.0,
                )
                return response.choices[0].delta.content
            except openai.OpenAIError as e:
                raise MlflowException("Failed to create completions") from e

        with pytest.raises(MlflowException, match="Failed to create completions"):
            create_completions("test")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "ERROR"

    assert len(trace.data.spans) == 2
    parent_span = trace.data.spans[0]
    assert parent_span.name == "create_completions"
    assert parent_span.inputs == {"text": "test"}
    assert parent_span.outputs is None
    assert parent_span.status.status_code == "ERROR"
    assert parent_span.events[0].name == "exception"
    assert parent_span.events[0].attributes["exception.type"] == "MlflowException"
    assert parent_span.events[0].attributes["exception.message"] == "Failed to create completions"

    child_span = trace.data.spans[1]
    assert child_span.name == "AsyncCompletions" if client._is_async else "Completions"
    assert child_span.inputs["messages"][0]["content"] == "test"
    assert child_span.outputs is None
    assert child_span.status.status_code == "ERROR"
    assert child_span.events[0].name == "exception"
    assert child_span.events[0].attributes["exception.type"] == "UnprocessableEntityError"

async def test_chat_completions_streaming_empty_choices(client):
    mlflow.openai.autolog()
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": EMPTY_CHOICES}],
        model="gpt-4o-mini",
        stream=True,
    )

    chunks = [chunk async for chunk in await stream] if client._is_async else list(stream)

    # Ensure the stream has a chunk with empty choices
    assert chunks[0].choices == []

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

async def test_chat_completions_streaming_with_list_content(client):
    # Test streaming with Databricks-style list content in chunks.
    mlflow.openai.autolog()
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": LIST_CONTENT}],
        model="gpt-4o-mini",
        stream=True,
    )

    chunks = [chunk async for chunk in await stream] if client._is_async else list(stream)

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == [{"type": "text", "text": "Hello"}]
    assert chunks[1].choices[0].delta.content == [{"type": "text", "text": " world"}]

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL

    # Verify the reconstructed message content is correct (text extracted from list)
    assert isinstance(span.outputs, dict)
    assert span.outputs["choices"][0]["message"]["content"] == "Hello world"

async def test_completions_autolog(client):
    mlflow.openai.autolog()

    response = client.completions.create(
        prompt="test",
        model="gpt-4o-mini",
        temperature=0,
    )

    if client._is_async:
        await response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.LLM
    assert span.inputs == {"prompt": "test", "model": "gpt-4o-mini", "temperature": 0}
    assert span.outputs["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert span.model_name == "gpt-4o-mini"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "openai"
    assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata

async def test_completions_autolog_streaming_empty_choices(client):
    mlflow.openai.autolog()
    stream = client.completions.create(
        prompt=EMPTY_CHOICES,
        model="gpt-4o-mini",
        stream=True,
    )

    chunks = [chunk async for chunk in await stream] if client._is_async else list(stream)

    # Ensure the stream has a chunk with empty choices
    assert chunks[0].choices == []

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

async def test_completions_autolog_streaming(client):
    mlflow.openai.autolog()

    stream = client.completions.create(
        prompt="test",
        model="gpt-4o-mini",
        temperature=0,
        stream=True,
    )
    if client._is_async:
        async for _ in await stream:
            pass
    else:
        for _ in stream:
            pass

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.LLM
    assert span.inputs == {
        "prompt": "test",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "stream": True,
    }
    assert span.outputs == "Hello world"  # aggregated string of streaming response

    stream_event_data = trace.data.spans[0].events

    assert stream_event_data[0].name == "mlflow.chunk.item.0"
    chunk_1 = json.loads(stream_event_data[0].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_1["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert chunk_1["choices"][0]["text"] == "Hello"
    assert stream_event_data[1].name == "mlflow.chunk.item.1"
    chunk_2 = json.loads(stream_event_data[1].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_2["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert chunk_2["choices"][0]["text"] == " world"

async def test_embeddings_autolog(client):
    mlflow.openai.autolog()

    response = client.embeddings.create(
        input="test",
        model="text-embedding-ada-002",
    )

    if client._is_async:
        await response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.EMBEDDING
    assert span.inputs == {"input": "test", "model": "text-embedding-ada-002"}
    assert span.outputs["data"][0]["embedding"] == list(range(1536))
    assert span.model_name == "text-embedding-ada-002"

    assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata

async def test_autolog_use_active_run_id(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]

    async def _call_create():
        response = client.chat.completions.create(messages=messages, model="gpt-4o-mini")
        if client._is_async:
            await response
        return response

    with mlflow.start_run() as run_1:
        await _call_create()

    with mlflow.start_run() as run_2:
        await _call_create()
        await _call_create()

    with mlflow.start_run() as run_3:
        mlflow.openai.autolog()
        await _call_create()

    traces = get_traces()[::-1]  # reverse order to sort by timestamp in ascending order
    assert len(traces) == 4

    assert traces[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_1.info.run_id
    assert traces[1].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_2.info.run_id
    assert traces[2].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_2.info.run_id
    assert traces[3].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_3.info.run_id

async def test_autolog_link_traces_to_loaded_model_chat_completions(client, completion_models):
    mlflow.openai.autolog()

    for model_info in completion_models:
        model_dict = mlflow.openai.load_model(model_info.model_uri)
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": f"test {model_info.model_id}"}],
            model=model_dict["model"],
            temperature=model_dict["temperature"],
        )
        if client._is_async:
            await resp

    traces = get_traces()
    assert len(traces) == len(completion_models)
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs["messages"][0]["content"] == f"test {model_id}"
        assert span.model_name == model_dict["model"]

async def test_autolog_link_traces_to_loaded_model_completions(client, completion_models):
    mlflow.openai.autolog()

    for model_info in completion_models:
        model_dict = mlflow.openai.load_model(model_info.model_uri)
        resp = client.completions.create(
            prompt=f"test {model_info.model_id}",
            model=model_dict["model"],
            temperature=model_dict["temperature"],
        )
        if client._is_async:
            await resp

    traces = get_traces()
    assert len(traces) == len(completion_models)
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[TraceMetadataKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs["prompt"] == f"test {model_id}"
        assert span.model_name == model_dict["model"]

def test_reconstruct_response_from_stream():
    from openai.types.responses import (
        ResponseOutputItemDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from mlflow.openai.autolog import _reconstruct_response_from_stream
    from mlflow.types.responses_helpers import OutputItem

    content1 = ResponseOutputText(annotations=[], text="Hello", type="output_text")
    content2 = ResponseOutputText(annotations=[], text=" world", type="output_text")

    message1 = ResponseOutputMessage(
        id="test-1", content=[content1], role="assistant", status="completed", type="message"
    )

    message2 = ResponseOutputMessage(
        id="test-2", content=[content2], role="assistant", status="completed", type="message"
    )

    chunk1 = ResponseOutputItemDoneEvent(
        item=message1, output_index=0, sequence_number=1, type="response.output_item.done"
    )

    chunk2 = ResponseOutputItemDoneEvent(
        item=message2, output_index=1, sequence_number=2, type="response.output_item.done"
    )

    chunks = [chunk1, chunk2]

    result = _reconstruct_response_from_stream(chunks)

    assert result.output == [
        OutputItem(**chunk1.item.to_dict()),
        OutputItem(**chunk2.item.to_dict()),
    ]


# --- tests/openai/test_openai_evaluate.py ---

def test_openai_pyfunc_evaluate(client):
    with mlflow.start_run() as run:
        model_info = mlflow.openai.log_model(
            "gpt-4o-mini",
            "chat.completions",
            name="model",
            messages=[{"role": "system", "content": "You are an MLflow expert."}],
        )

        evaluate(
            model_info.model_uri,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )
    assert len(get_traces()) == 2
    assert run.info.run_id == get_traces()[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN]

def test_openai_evaluate_should_not_log_traces_when_disabled(client, globally_disabled):
    if globally_disabled:
        mlflow.autolog(disable=True)
    else:
        mlflow.openai.autolog(disable=True)

    def model(inputs):
        return [
            client.chat.completions
            .create(
                messages=[{"role": "user", "content": question}],
                model="gpt-4o-mini",
                temperature=0.0,
            )
            .choices[0]
            .message.content
            for question in inputs["inputs"]
        ]

    with mlflow.start_run():
        evaluate(
            model,
            data=_EVAL_DATA,
            targets="ground_truth",
            extra_metrics=[mlflow.metrics.exact_match()],
        )

    assert len(get_traces()) == 0


# --- tests/openai/test_openai_model_export.py ---

def test_log_model():
    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            "gpt-4o-mini",
            "chat.completions",
            name="model",
            temperature=0.9,
            messages=[{"role": "system", "content": "You are an MLflow expert."}],
        )

    loaded_model = mlflow.openai.load_model(model_info.model_uri)
    assert loaded_model["model"] == "gpt-4o-mini"
    assert loaded_model["task"] == "chat.completions"
    assert loaded_model["temperature"] == 0.9
    assert loaded_model["messages"] == [{"role": "system", "content": "You are an MLflow expert."}]

def test_chat_single_variable(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[{"role": "user", "content": "{x}"}],
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "x": [
            "a",
            "b",
        ]
    })
    expected_output = [
        [{"content": "a", "role": "user"}],
        [{"content": "b", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"x": "a"},
        {"x": "b"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        "a",
        "b",
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

def test_completion_single_variable(tmp_path):
    mlflow.openai.save_model(
        model="text-davinci-003",
        task=completions(),
        path=tmp_path,
        prompt="Say {text}",
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "x": [
            "this is a test",
            "this is another test",
        ]
    })
    expected_output = ["Say this is a test", "Say this is another test"]
    assert model.predict(data) == expected_output

    data = [
        {"x": "this is a test"},
        {"x": "this is another test"},
    ]
    assert model.predict(data) == expected_output

    data = [
        "this is a test",
        "this is another test",
    ]
    assert model.predict(data) == expected_output

def test_chat_multiple_variables(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[{"role": "user", "content": "{x} {y}"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string", "required": True},
        {"name": "y", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "x": [
            "a",
            "b",
        ],
        "y": [
            "c",
            "d",
        ],
    })
    expected_output = [
        [{"content": "a c", "role": "user"}],
        [{"content": "b d", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"x": "a", "y": "c"},
        {"x": "b", "y": "d"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

def test_chat_role_content(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[{"role": "{role}", "content": "{content}"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "content", "type": "string", "required": True},
        {"name": "role", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "role": [
            "system",
            "user",
        ],
        "content": [
            "c",
            "d",
        ],
    })
    expected_output = [
        [{"content": "c", "role": "system"}],
        [{"content": "d", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

def test_completion_multiple_variables(tmp_path):
    mlflow.openai.save_model(
        model="text-davinci-003",
        task=completions(),
        path=tmp_path,
        prompt="Say {x} and {y}",
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string", "required": True},
        {"name": "y", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "x": [
            "a",
            "b",
        ],
        "y": [
            "c",
            "d",
        ],
    })
    expected_output = ["Say a and c", "Say b and d"]
    assert model.predict(data) == expected_output

    data = [
        {"x": "a", "y": "c"},
        {"x": "b", "y": "d"},
    ]
    assert model.predict(data) == expected_output

def test_chat_multiple_messages(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[
            {"role": "user", "content": "{x}"},
            {"role": "user", "content": "{y}"},
        ],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "x", "type": "string", "required": True},
        {"name": "y", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "x": [
            "a",
            "b",
        ],
        "y": [
            "c",
            "d",
        ],
    })
    expected_output = [
        [{"content": "a", "role": "user"}, {"content": "c", "role": "user"}],
        [{"content": "b", "role": "user"}, {"content": "d", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"x": "a", "y": "c"},
        {"x": "b", "y": "d"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

def test_chat_no_variables(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[{"role": "user", "content": "a"}],
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "content": ["b", "c"],
    })
    expected_output = [
        [{"content": "a", "role": "user"}, {"content": "b", "role": "user"}],
        [{"content": "a", "role": "user"}, {"content": "c", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"content": "b"},
        {"content": "c"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        "b",
        "c",
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

def test_completion_no_variable(tmp_path):
    mlflow.openai.save_model(
        model="text-davinci-003",
        task=completions(),
        path=tmp_path,
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "x": [
            "this is a test",
            "this is another test",
        ]
    })
    expected_output = ["this is a test", "this is another test"]
    assert model.predict(data) == expected_output

    data = [
        {"x": "this is a test"},
        {"x": "this is another test"},
    ]
    assert model.predict(data) == expected_output

    data = [
        "this is a test",
        "this is another test",
    ]
    assert model.predict(data) == expected_output

def test_chat_no_messages(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
    )
    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "content": ["b", "c"],
    })
    expected_output = [
        [{"content": "b", "role": "user"}],
        [{"content": "c", "role": "user"}],
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        {"content": "b"},
        {"content": "c"},
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

    data = [
        "b",
        "c",
    ]
    assert list(map(json.loads, model.predict(data))) == expected_output

def test_invalid_messages(tmp_path, messages):
    with pytest.raises(
        mlflow.MlflowException,
        match="it must be a list of dictionaries with keys 'role' and 'content'",
    ):
        mlflow.openai.save_model(
            model="gpt-4o-mini",
            task=chat_completions(),
            path=tmp_path,
            messages=messages,
        )

def test_task_argument_accepts_class(tmp_path):
    mlflow.openai.save_model(model="gpt-4o-mini", task=chat_completions(), path=tmp_path)
    loaded_model = mlflow.openai.load_model(tmp_path)
    assert loaded_model["task"] == "chat.completions"

def test_model_argument_accepts_retrieved_model(tmp_path):
    model = openai.Model.retrieve("gpt-4o-mini")
    mlflow.openai.save_model(model=model, task=chat_completions(), path=tmp_path)
    loaded_model = mlflow.openai.load_model(tmp_path)
    assert loaded_model["model"] == "gpt-4o-mini"

def test_spark_udf_chat(tmp_path, spark):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task="chat.completions",
        path=tmp_path,
        messages=[
            {"role": "user", "content": "{x} {y}"},
        ],
    )
    udf = mlflow.pyfunc.spark_udf(spark, tmp_path, result_type="string")
    df = spark.createDataFrame(
        [
            ("a", "b"),
            ("c", "d"),
        ],
        ["x", "y"],
    )
    df = df.withColumn("z", udf())
    pdf = df.toPandas()
    assert list(map(json.loads, pdf["z"])) == [
        [{"content": "a b", "role": "user"}],
        [{"content": "c d", "role": "user"}],
    ]

def test_embeddings(tmp_path):
    mlflow.openai.save_model(
        model="text-embedding-ada-002",
        task=embeddings(),
        path=tmp_path,
    )

    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [{"type": "string", "required": True}]
    assert model.signature.outputs.to_dict() == [
        {"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": (-1,)}}
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({"text": ["a", "b"]})
    preds = model.predict(data)
    assert list(map(len, preds)) == [1536, 1536]

    data = pd.DataFrame({"text": ["a"] * 100})
    preds = model.predict(data)
    assert list(map(len, preds)) == [1536] * 100

def test_embeddings_pyfunc_server_and_score():
    df = pd.DataFrame({"text": ["a", "b"]})
    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            "text-embedding-ada-002",
            embeddings(),
            name="model",
            input_example=df,
        )
    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    expected = mlflow.pyfunc.load_model(model_info.model_uri).predict(df)
    actual = pd.DataFrame(data=json.loads(resp.content.decode("utf-8")))
    pd.testing.assert_frame_equal(actual, pd.DataFrame({"predictions": expected}))

def test_inference_params(tmp_path):
    mlflow.openai.save_model(
        model="text-embedding-ada-002",
        task=embeddings(),
        path=tmp_path,
        signature=ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
            params=ParamSchema([ParamSpec(name="batch_size", dtype="long", default=16)]),
        ),
    )

    model_info = mlflow.models.Model.load(tmp_path)
    assert (
        len([p for p in model_info.signature.params if p.name == "batch_size" and p.default == 16])
        == 1
    )

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({"text": ["a", "b"]})
    preds = model.predict(data, params={"batch_size": 5})
    assert list(map(len, preds)) == [1536, 1536]

def test_inference_params_overlap(tmp_path):
    with pytest.raises(mlflow.MlflowException, match=r"any of \['prefix'\] as parameters"):
        mlflow.openai.save_model(
            model="text-davinci-003",
            task=completions(),
            path=tmp_path,
            prefix="Classify the following text's sentiment:",
            signature=ModelSignature(
                inputs=Schema([ColSpec(type="string", name=None)]),
                outputs=Schema([ColSpec(type="string", name=None)]),
                params=ParamSchema([ParamSpec(name="prefix", default=None, dtype="string")]),
            ),
        )

def test_multimodal_messages(tmp_path):
    # Test multimodal content with variable placeholders
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "{system_prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{image_base64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
    )

    model = mlflow.models.Model.load(tmp_path)
    assert model.signature.inputs.to_dict() == [
        {"name": "image_base64", "type": "string", "required": True},
        {"name": "system_prompt", "type": "string", "required": True},
    ]
    assert model.signature.outputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({
        "system_prompt": ["Analyze this image"],
        "image_base64": [
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ],
    })

    expected_output = [
        [
            {
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                            ),
                            "detail": "low",
                        },
                    },
                ],
                "role": "user",
            }
        ]
    ]

    assert list(map(json.loads, model.predict(data))) == expected_output

def test_multimodal_messages_no_variables(tmp_path):
    mlflow.openai.save_model(
        model="gpt-4o-mini",
        task=chat_completions(),
        path=tmp_path,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123", "detail": "low"},
                    },
                ],
            }
        ],
    )

    model = mlflow.models.Model.load(tmp_path)
    # Should add default content variable since no variables found
    assert model.signature.inputs.to_dict() == [
        {"type": "string", "required": True},
    ]

    model = mlflow.pyfunc.load_model(tmp_path)
    data = pd.DataFrame({"content": ["Additional context"]})

    expected_output = [
        [
            {
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123", "detail": "low"},
                    },
                ],
                "role": "user",
            },
            {"content": "Additional context", "role": "user"},
        ]
    ]

    assert list(map(json.loads, model.predict(data))) == expected_output


# --- tests/openai/test_openai_responses_autolog.py ---

async def test_responses_stream_autolog(client):
    mlflow.openai.autolog()

    response = client.responses.create(
        input="Hello",
        model="gpt-4o",
        stream=True,
    )

    if client._is_async:
        async for _ in await response:
            pass
    else:
        for _ in response:
            pass

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.outputs["id"] == "responses-123"
    # "logprobs" is only returned from certain version of OpenAI SDK
    span.outputs["output"][0]["content"][0].pop("logprobs", None)
    assert span.outputs["output"][0]["content"] == [
        {
            "text": "Dummy output",
            "annotations": None,
            "type": "output_text",
        }
    ]
    assert span.attributes["model"] == "gpt-4o"
    assert span.attributes["stream"] is True

    # Token usage should be aggregated correctly
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 36,
        TokenUsageKey.OUTPUT_TOKENS: 87,
        TokenUsageKey.TOTAL_TOKENS: 123,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 0,
    }


# --- tests/pytorch/test_pytorch_model_export.py ---

def test_log_model(sequential_model, data, sequential_predicted):
    try:
        artifact_path = "pytorch"
        model_info = mlflow.pytorch.log_model(sequential_model, name=artifact_path)

        sequential_model_loaded = mlflow.pytorch.load_model(model_uri=model_info.model_uri)
        test_predictions = _predict(sequential_model_loaded, data)
        np.testing.assert_array_equal(test_predictions, sequential_predicted)
    finally:
        mlflow.end_run()

def test_save_and_load_model(sequential_model, model_path, data, sequential_predicted):
    mlflow.pytorch.save_model(sequential_model, model_path)

    # Loading pytorch model
    sequential_model_loaded = mlflow.pytorch.load_model(model_path)
    np.testing.assert_array_equal(_predict(sequential_model_loaded, data), sequential_predicted)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_array_almost_equal(
        pyfunc_loaded.predict(data[0]).values[:, 0], sequential_predicted, decimal=4
    )

def test_pyfunc_model_works_with_np_input_type(
    sequential_model, model_path, data, sequential_predicted
):
    mlflow.pytorch.save_model(sequential_model, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    # predict works with dataframes
    df_result = pyfunc_loaded.predict(data[0])
    assert type(df_result) == pd.DataFrame
    np.testing.assert_array_almost_equal(df_result.values[:, 0], sequential_predicted, decimal=4)

    # predict works with numpy ndarray
    np_result = pyfunc_loaded.predict(data[0].values.astype(np.float32))
    assert type(np_result) == np.ndarray
    np.testing.assert_array_almost_equal(np_result[:, 0], sequential_predicted, decimal=4)

    # predict does not work with lists
    with pytest.raises(
        TypeError, match="The PyTorch flavor does not support List or Dict input types"
    ):
        pyfunc_loaded.predict([1, 2, 3, 4])

    # predict does not work with scalars
    with pytest.raises(TypeError, match="Input data should be pandas.DataFrame or numpy.ndarray"):
        pyfunc_loaded.predict(4)

def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    sequential_model, pytorch_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            sequential_model,
            name=artifact_path,
            conda_env=pytorch_custom_env,
        )
        model_path = _download_artifact_from_uri(model_info.model_uri)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pytorch_custom_env

    with open(pytorch_custom_env) as f:
        pytorch_custom_env_text = f.read()
    with open(saved_conda_env_path) as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == pytorch_custom_env_text

def test_model_log_persists_requirements_in_mlflow_model_directory(
    sequential_model, pytorch_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            sequential_model,
            name=artifact_path,
            conda_env=pytorch_custom_env,
        )
        model_path = _download_artifact_from_uri(model_info.model_uri)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(pytorch_custom_env, saved_pip_req_path)

def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sequential_model,
):
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(sequential_model, name="model")

    _assert_pip_requirements(model_info.model_uri, mlflow.pytorch.get_default_pip_requirements())

def test_pyfunc_model_serving_with_module_scoped_subclassed_model_and_default_conda_env(
    module_scoped_subclassed_model, data
):
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            module_scoped_subclassed_model,
            name="pytorch_model",
            code_paths=[__file__],
            input_example=data[0],
        )

    inference_payload = load_serving_example(model_info.model_uri)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200

    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=module_scoped_subclassed_model, data=data),
        decimal=4,
    )

def test_pyfunc_model_serving_with_main_scoped_subclassed_model_and_custom_pickle_module(
    main_scoped_subclassed_model, data
):
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            main_scoped_subclassed_model,
            name="pytorch_model",
            pickle_module=mlflow_pytorch_pickle_module,
            input_example=data[0],
        )

    inference_payload = load_serving_example(model_info.model_uri)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200

    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=main_scoped_subclassed_model, data=data),
        decimal=4,
    )

def test_load_model_succeeds_with_dependencies_specified_via_code_paths(
    module_scoped_subclassed_model, model_path, data
):
    # Save a PyTorch model whose class is defined in the current test suite. Because the
    # `tests` module is not available when the model is deployed for local scoring, we include
    # the test suite file as a code dependency
    mlflow.pytorch.save_model(
        path=model_path,
        pytorch_model=module_scoped_subclassed_model,
        code_paths=[__file__],
    )

    # Define a custom pyfunc model that loads a PyTorch model artifact using
    # `mlflow.pytorch.load_model`
    class TorchValidatorModel(pyfunc.PythonModel):
        def load_context(self, context):
            self.pytorch_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])

        def predict(self, context, model_input, params=None):
            with torch.no_grad():
                input_tensor = torch.from_numpy(model_input.values.astype(np.float32))
                output_tensor = self.pytorch_model(input_tensor)
                return pd.DataFrame(output_tensor.numpy())

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        model_info = pyfunc.log_model(
            pyfunc_artifact_path,
            python_model=TorchValidatorModel(),
            artifacts={"pytorch_model": model_path},
            input_example=data[0],
            # save file into code_paths, otherwise after first model loading (happens when
            # validating input_example) then we can not load the model again
            code_paths=[__file__],
        )

    # Deploy the custom pyfunc model and ensure that it is able to successfully load its
    # constituent PyTorch model via `mlflow.pytorch.load_model`
    inference_payload = load_serving_example(model_info.model_uri)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200

    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=module_scoped_subclassed_model, data=data),
        decimal=4,
    )

def test_load_pyfunc_succeeds_when_data_is_model_file_instead_of_directory(
    module_scoped_subclassed_model, model_path, data
):
    """
    This test verifies that PyTorch models saved in older versions of MLflow are loaded successfully
    by ``mlflow.pytorch.load_model``. The ``data`` path associated with these older models is
    serialized PyTorch model file, as opposed to the current format: a directory containing a
    serialized model file and pickle module information.
    """
    mlflow.pytorch.save_model(path=model_path, pytorch_model=module_scoped_subclassed_model)

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    pyfunc_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME)
    assert pyfunc_conf is not None
    model_data_path = os.path.join(model_path, pyfunc_conf[pyfunc.DATA])
    assert os.path.exists(model_data_path)
    assert mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME in os.listdir(model_data_path)
    pyfunc_conf[pyfunc.DATA] = os.path.join(
        model_data_path, mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME
    )
    model_conf.save(model_conf_path)

    loaded_pyfunc = pyfunc.load_model(model_path)

    np.testing.assert_array_almost_equal(
        loaded_pyfunc.predict(data[0]),
        pd.DataFrame(_predict(model=module_scoped_subclassed_model, data=data)),
        decimal=4,
    )

def test_load_model_succeeds_when_data_is_model_file_instead_of_directory(
    module_scoped_subclassed_model, model_path, data
):
    """
    This test verifies that PyTorch models saved in older versions of MLflow are loaded successfully
    by ``mlflow.pytorch.load_model``. The ``data`` path associated with these older models is
    serialized PyTorch model file, as opposed to the current format: a directory containing a
    serialized model file and pickle module information.
    """
    artifact_path = "pytorch_model"
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(module_scoped_subclassed_model, name=artifact_path)
        model_path = _download_artifact_from_uri(model_info.model_uri)

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    pyfunc_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME)
    assert pyfunc_conf is not None
    model_data_path = os.path.join(model_path, pyfunc_conf[pyfunc.DATA])
    assert os.path.exists(model_data_path)
    assert mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME in os.listdir(model_data_path)
    pyfunc_conf[pyfunc.DATA] = os.path.join(
        model_data_path, mlflow.pytorch._SERIALIZED_TORCH_MODEL_FILE_NAME
    )
    model_conf.save(model_conf_path)

    loaded_pyfunc = pyfunc.load_model(model_path)

    np.testing.assert_array_almost_equal(
        loaded_pyfunc.predict(data[0]),
        pd.DataFrame(_predict(model=module_scoped_subclassed_model, data=data)),
        decimal=4,
    )

def test_pyfunc_serve_and_score(data):
    model = torch.nn.Linear(4, 1)
    train_model(model=model, data=data)

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(model, name="model", input_example=data[0])

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    from mlflow.deployments import PredictionsResponse

    scores = PredictionsResponse.from_json(resp.content).get_predictions()
    np.testing.assert_array_almost_equal(scores.values[:, 0], _predict(model=model, data=data))

def test_pyfunc_serve_and_score_transformers():
    from transformers import BertConfig, BertModel

    from mlflow.deployments import PredictionsResponse

    class MyBertModel(BertModel):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs).last_hidden_state

    model = MyBertModel(
        BertConfig(
            vocab_size=16,
            hidden_size=2,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=2,
        )
    )
    model.eval()

    input_ids = model.dummy_inputs["input_ids"]

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            model, name="model", input_example=np.array(input_ids.tolist())
        )

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )

    scores = PredictionsResponse.from_json(resp.content.decode("utf-8")).get_predictions(
        predictions_format="ndarray"
    )
    assert_array_almost_equal(scores, model(input_ids).detach().numpy(), rtol=1e-6)

def test_extra_files_log_model(create_extra_files, sequential_model):
    extra_files, contents_expected = create_extra_files
    with mlflow.start_run():
        mlflow.pytorch.log_model(sequential_model, name="models", extra_files=extra_files)

        model_uri = "runs:/{run_id}/{model_path}".format(
            run_id=mlflow.active_run().info.run_id, model_path="models"
        )
        with TempDir(remove_on_exit=True) as tmp:
            model_path = _download_artifact_from_uri(model_uri, tmp.path())
            model_config_path = os.path.join(model_path, "MLmodel")
            model_config = Model.load(model_config_path)
            flavor_config = model_config.flavors["pytorch"]

            assert "extra_files" in flavor_config
            loaded_extra_files = flavor_config["extra_files"]

            for loaded_extra_file, content_expected in zip(loaded_extra_files, contents_expected):
                assert "path" in loaded_extra_file
                extra_file_path = os.path.join(model_path, loaded_extra_file["path"])
                with open(extra_file_path) as fp:
                    assert fp.read() == content_expected

def test_log_model_invalid_extra_file_path(sequential_model):
    with (
        mlflow.start_run(),
        pytest.raises(MlflowException, match="No such artifact: 'non_existing_file.txt'"),
    ):
        mlflow.pytorch.log_model(
            sequential_model,
            name="models",
            extra_files=["non_existing_file.txt"],
        )

def test_log_model_invalid_extra_file_type(sequential_model):
    with (
        mlflow.start_run(),
        pytest.raises(TypeError, match="Extra files argument should be a list"),
    ):
        mlflow.pytorch.log_model(
            sequential_model,
            name="models",
            extra_files="non_existing_file.txt",
        )

def test_save_state_dict(sequential_model, model_path, data):
    state_dict = sequential_model.state_dict()
    mlflow.pytorch.save_state_dict(state_dict, model_path)

    loaded_state_dict = mlflow.pytorch.load_state_dict(model_path)
    assert state_dict_equal(loaded_state_dict, state_dict)
    model = get_sequential_model()
    model.load_state_dict(loaded_state_dict)
    np.testing.assert_array_almost_equal(
        _predict(model, data),
        _predict(sequential_model, data),
        decimal=4,
    )

def test_log_state_dict(sequential_model, data):
    artifact_path = "model"
    state_dict = sequential_model.state_dict()
    with mlflow.start_run():
        mlflow.pytorch.log_state_dict(state_dict, artifact_path)
        state_dict_uri = mlflow.get_artifact_uri(artifact_path)

    loaded_state_dict = mlflow.pytorch.load_state_dict(state_dict_uri)
    assert state_dict_equal(loaded_state_dict, state_dict)
    model = get_sequential_model()
    model.load_state_dict(loaded_state_dict)
    np.testing.assert_array_almost_equal(
        _predict(model, data),
        _predict(sequential_model, data),
        decimal=4,
    )

def test_model_log_with_metadata(sequential_model):
    artifact_path = "model"

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            sequential_model,
            name=artifact_path,
            metadata={"metadata_key": "metadata_value"},
        )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"

def test_model_log_with_signature_inference(sequential_model, data):
    artifact_path = "model"
    example_ = data[0].head(3).values.astype(np.float32)

    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(
            sequential_model, name=artifact_path, input_example=example_
        )

    assert model_info.signature == ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 4))]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 1))]),
    )
    inference_payload = load_serving_example(model_info.model_uri)
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200
    deployed_model_preds = pd.DataFrame(json.loads(response.content)["predictions"])
    np.testing.assert_array_almost_equal(
        deployed_model_preds.values[:, 0],
        _predict(model=sequential_model, data=(data[0].head(3), data[1].head(3))),
        decimal=4,
    )

def test_passing_params_to_model(data):
    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)

        def forward(self, x, y):
            if not torch.is_tensor(x):
                x = torch.from_numpy(x)
            y = torch.tensor(y)
            combined = x * y
            return self.linear(combined)

    model = CustomModel()
    x = np.random.randn(8, 4).astype(np.float32)

    signature = mlflow.models.infer_signature(x, None, {"y": 1})
    with mlflow.start_run():
        model_info = mlflow.pytorch.log_model(model, name="model", signature=signature)

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with torch.no_grad():
        np.testing.assert_array_almost_equal(pyfunc_model.predict(x), model(x, 1), decimal=4)
        np.testing.assert_array_almost_equal(
            pyfunc_model.predict(x, {"y": 2}), model(x, 2), decimal=4
        )

def test_log_model_with_datetime_input():
    df = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=5, freq="D"),
        "x": np.random.uniform(20, 30, 5),
        "y": np.random.uniform(2, 4, 5),
        "z": np.random.uniform(0, 10, 5),
    })
    model = get_sequential_model()
    model_info = mlflow.pytorch.log_model(model, name="pytorch", input_example=df)
    assert model_info.signature.inputs.inputs[0].type == DataType.datetime
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with torch.no_grad():
        input_tensor = torch.from_numpy(df.to_numpy(dtype=np.float32))
        expected_result = model(input_tensor)
    with torch.no_grad():
        np.testing.assert_array_almost_equal(pyfunc_model.predict(df), expected_result, decimal=4)

def test_save_and_load_exported_model(sequential_model, model_path, data, sequential_predicted):
    input_example = data[0].to_numpy(dtype=np.float32)

    mlflow.pytorch.save_model(
        sequential_model,
        model_path,
        serialization_format="pt2",
        input_example=input_example,
    )

    # Loading pytorch model
    sequential_model_loaded = mlflow.pytorch.load_model(model_path)
    np.testing.assert_array_equal(_predict(sequential_model_loaded, data), sequential_predicted)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_array_almost_equal(
        pyfunc_loaded.predict(input_example)[:, 0], sequential_predicted, decimal=4
    )


# --- tests/sagemaker/test_batch_deployment.py ---

def test_deploy_cli_creates_sagemaker_transform_job_and_s3_resources_with_expected_names_from_local(
    pretrained_model, sagemaker_client
):
    job_name = "test-job"
    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        mfscli.commands,
        [
            "deploy-transform-job",
            "--job-name",
            job_name,
            "--model-uri",
            pretrained_model.model_uri,
            "--input-data-type",
            "Some Data Type",
            "--input-uri",
            "Some Input Uri",
            "--content-type",
            "Some Content Type",
            "--output-path",
            "Some Output Path",
            "--archive",
        ],
    )
    assert result.exit_code == 0

    region_name = sagemaker_client.meta.region_name
    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    model_name = transform_job_description["ModelName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert job_name in [
        transform_job["TransformJobName"]
        for transform_job in sagemaker_client.list_transform_jobs()["TransformJobSummaries"]
    ]

def test_deploy_cli_creates_sagemaker_transform_job_and_s3_resources_with_expected_names_from_s3(
    pretrained_model, sagemaker_client
):
    local_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    artifact_path = "model"
    region_name = sagemaker_client.meta.region_name
    default_bucket = mfs._get_default_s3_bucket(region_name)
    s3_artifact_repo = S3ArtifactRepository(f"s3://{default_bucket}")
    s3_artifact_repo.log_artifacts(local_model_path, artifact_path=artifact_path)
    model_s3_uri = f"s3://{default_bucket}/{pretrained_model.model_path}"

    job_name = "test-job"
    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        mfscli.commands,
        [
            "deploy-transform-job",
            "--job-name",
            job_name,
            "--model-uri",
            model_s3_uri,
            "--input-data-type",
            "Some Data Type",
            "--input-uri",
            "Some Input Uri",
            "--content-type",
            "Some Content Type",
            "--output-path",
            "Some Output Path",
            "--archive",
        ],
    )
    assert result.exit_code == 0

    region_name = sagemaker_client.meta.region_name
    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    model_name = transform_job_description["ModelName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert job_name in [
        transform_job["TransformJobName"]
        for transform_job in sagemaker_client.list_transform_jobs()["TransformJobSummaries"]
    ]

def test_deploy_in_synchronous_mode_waits_for_transform_job_creation_to_complete_before_returning(
    pretrained_model, sagemaker_client
):
    transform_job_creation_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_transform_job_update_latency(
        transform_job_creation_latency
    )

    job_name = "test-job"
    deployment_start_time = time.time()
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
        synchronous=True,
    )
    deployment_end_time = time.time()

    assert (deployment_end_time - deployment_start_time) >= transform_job_creation_latency
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    assert transform_job_description["TransformJobStatus"] == TransformJob.STATUS_COMPLETED


# --- tests/sagemaker/test_sagemaker_deployment_client.py ---

def test_update_deployment_in_replace_mode_with_archiving_does_not_delete_resources(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    region_name = sagemaker_client.meta.region_name
    sagemaker_backend = get_sagemaker_backend(region_name)
    sagemaker_backend.set_endpoint_update_latency(5)

    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )

    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    object_names_before_replacement = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    endpoint_configs_before_replacement = [
        config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    ]
    models_before_replacement = [
        model["ModelName"] for model in sagemaker_client.list_models()["Models"]
    ]

    model_uri = f"runs:/{pretrained_model.run_id}/{pretrained_model.model_path}"
    sk_model = mlflow.sklearn.load_model(model_uri=model_uri)
    new_artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model, name=new_artifact_path)
        new_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{new_artifact_path}"
    sagemaker_deployment_client.update_deployment(
        name=name,
        model_uri=new_model_uri,
        config={"mode": mfs.DEPLOYMENT_MODE_REPLACE, "archive": True, "synchronous": True},
    )

    object_names_after_replacement = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    endpoint_configs_after_replacement = [
        config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    ]
    models_after_replacement = [
        model["ModelName"] for model in sagemaker_client.list_models()["Models"]
    ]
    assert all(
        object_name in object_names_after_replacement
        for object_name in object_names_before_replacement
    )
    assert all(
        endpoint_config in endpoint_configs_after_replacement
        for endpoint_config in endpoint_configs_before_replacement
    )
    assert all(model in models_after_replacement for model in models_before_replacement)

def test_deploy_cli_updates_sagemaker_and_s3_resources_in_replace_mode(
    pretrained_model, sagemaker_client
):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "update",
            "--target",
            f"sagemaker:/{region_name}",
            "--name",
            app_name,
            "--model-uri",
            pretrained_model.model_uri,
        ],
    )
    assert result.exit_code == 0

    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert any(
        app_name in config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]
    expected_model_environment = {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }
    if os.environ.get("http_proxy") is not None:
        expected_model_environment.update({"http_proxy": os.environ["http_proxy"]})

    if os.environ.get("https_proxy") is not None:
        expected_model_environment.update({"https_proxy": os.environ["https_proxy"]})

    if os.environ.get("no_proxy") is not None:
        expected_model_environment.update({"no_proxy": os.environ["no_proxy"]})

    assert model_environment == expected_model_environment

def test_deploy_cli_updates_sagemaker_and_s3_resources_in_add_mode(
    pretrained_model, sagemaker_client
):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "update",
            "--target",
            f"sagemaker:/{region_name}",
            "--name",
            app_name,
            "--model-uri",
            pretrained_model.model_uri,
            "--config",
            f"mode={mfs.DEPLOYMENT_MODE_ADD}",
        ],
    )
    assert result.exit_code == 0

    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 2

def test_deploy_cli_deletes_sagemaker_deployment(pretrained_model, sagemaker_client):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "delete",
            "--target",
            "sagemaker",
            "--name",
            app_name,
            "--config",
            f"region_name={region_name}",
        ],
    )
    assert result.exit_code == 0

    response = sagemaker_client.list_endpoints()
    assert len(response["Endpoints"]) == 0

def test_deploy_cli_gets_sagemaker_deployment(pretrained_model, sagemaker_client):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "get",
            "--target",
            f"sagemaker:/{region_name}",
            "--name",
            app_name,
        ],
    )

    assert result.exit_code == 0

def test_deploy_cli_list_sagemaker_deployments(pretrained_model, sagemaker_client):
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli("test-app-1", pretrained_model.model_uri, region_name)
    create_sagemaker_deployment_through_cli("test-app-2", pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "list",
            "--target",
            f"sagemaker:/{region_name}",
        ],
    )

    assert result.exit_code == 0

def test_truncate_name():
    assert mfs._truncate_name("a" * 64, 63) == "a" * 30 + "---" + "a" * 30
    assert mfs._truncate_name("a" * 10, 63) == "a" * 10
    assert mfs._truncate_name("abcdefghijklmnopqrst", 10) == "abc---qrst"


# --- tests/server/jobs/test_jobs.py ---

def test_start_job_is_atomic(tmp_path: Path, workspaces_enabled):
    backend_store_uri = f"sqlite:///{tmp_path / 'test.db'}"
    store_cls = WorkspaceAwareSqlAlchemyJobStore if workspaces_enabled else SqlAlchemyJobStore
    store = store_cls(backend_store_uri)

    job = store.create_job("test.function", '{"param": "value"}')
    assert job.status == JobStatus.PENDING

    results = []

    def try_start_job() -> str:
        try:
            store.start_job(job.job_id)
            return "success"
        except MlflowException:
            return "failed"

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=5, thread_name_prefix="test-concurrent-jobs"
    ) as executor:
        futures = [executor.submit(try_start_job) for _ in range(5)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert results.count("success") == 1
    assert results.count("failed") == 4

    final_job = store.get_job(job.job_id)
    assert final_job.status == JobStatus.RUNNING

def test_delete_jobs_only_deletes_finalized(tmp_path: Path):
    backend_store_uri = f"sqlite:///{tmp_path / 'test.db'}"
    store = SqlAlchemyJobStore(backend_store_uri)

    pending_job = store.create_job("pending_job", "{}")
    assert pending_job.status == JobStatus.PENDING

    running_job = store.create_job("running_job", "{}")
    store.start_job(running_job.job_id)
    running_job = store.get_job(running_job.job_id)
    assert running_job.status == JobStatus.RUNNING

    succeeded_job = store.create_job("succeeded_job", "{}")
    store.start_job(succeeded_job.job_id)
    store.finish_job(succeeded_job.job_id, "result")
    succeeded_job = store.get_job(succeeded_job.job_id)
    assert succeeded_job.status == JobStatus.SUCCEEDED

    failed_job = store.create_job("failed_job", "{}")
    store.start_job(failed_job.job_id)
    store.fail_job(failed_job.job_id, "error")
    failed_job = store.get_job(failed_job.job_id)
    assert failed_job.status == JobStatus.FAILED

    timeout_job = store.create_job("timeout_job", "{}")
    store.start_job(timeout_job.job_id)
    store.mark_job_timed_out(timeout_job.job_id)
    timeout_job = store.get_job(timeout_job.job_id)
    assert timeout_job.status == JobStatus.TIMEOUT

    canceled_job = store.create_job("canceled_job", "{}")
    store.cancel_job(canceled_job.job_id)
    canceled_job = store.get_job(canceled_job.job_id)
    assert canceled_job.status == JobStatus.CANCELED

    deleted_ids = store.delete_jobs()

    # Should only delete finalized jobs
    assert len(deleted_ids) == 4
    assert succeeded_job.job_id in deleted_ids
    assert failed_job.job_id in deleted_ids
    assert timeout_job.job_id in deleted_ids
    assert canceled_job.job_id in deleted_ids

    # Non-finalized jobs should still exist
    assert store.get_job(pending_job.job_id).status == JobStatus.PENDING
    assert store.get_job(running_job.job_id).status == JobStatus.RUNNING

    # Finalized jobs should be deleted
    with pytest.raises(MlflowException, match=r"Job .+ not found"):
        store.get_job(succeeded_job.job_id)
    with pytest.raises(MlflowException, match=r"Job .+ not found"):
        store.get_job(failed_job.job_id)
    with pytest.raises(MlflowException, match=r"Job .+ not found"):
        store.get_job(timeout_job.job_id)
    with pytest.raises(MlflowException, match=r"Job .+ not found"):
        store.get_job(canceled_job.job_id)


# --- tests/tensorflow/test_keras_model_export.py ---

def test_model_save_load(build_model, save_format, model_path, data):
    x, _ = data
    keras_model = build_model(data)
    if build_model == get_tf_keras_model:
        model_path = os.path.join(model_path, "tf")
    else:
        model_path = os.path.join(model_path, "plain")
    expected = keras_model.predict(x)
    kwargs = {"save_format": save_format} if save_format else {}
    mlflow.tensorflow.save_model(keras_model, path=model_path, keras_model_kwargs=kwargs)
    # Loading Keras model
    model_loaded = mlflow.tensorflow.load_model(model_path)
    # When saving as SavedModel, we actually convert the model
    # to a slightly different format, so we cannot assume it is
    # exactly the same.
    if save_format != "tf":
        assert type(keras_model) == type(model_loaded)
    np.testing.assert_allclose(model_loaded.predict(x), expected, rtol=1e-5)
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_allclose(pyfunc_loaded.predict(x), expected, rtol=1e-5)

def test_pyfunc_serve_and_score(data):
    x, _ = data
    model = get_model(data)
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(model, name="model", input_example=x)
    expected = model.predict(x)
    inference_payload = load_serving_example(model_info.model_uri)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    actual_scoring_response = (
        PredictionsResponse
        .from_json(scoring_response.content.decode("utf-8"))
        .get_predictions()
        .values.astype(np.float32)
    )
    np.testing.assert_allclose(actual_scoring_response, expected, rtol=1e-5)

def test_score_model_as_spark_udf(data):
    x, _ = data
    model = get_model(data)
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(model, name="model")
    expected = model.predict(x)
    x_df = pd.DataFrame(x, columns=["0", "1", "2", "3"])
    spark_udf_preds = score_model_as_udf(
        model_uri=model_info.model_uri, pandas_df=x_df, result_type="float"
    )
    np.testing.assert_allclose(
        np.array(spark_udf_preds), expected.reshape(len(spark_udf_preds)), rtol=1e-5
    )

def test_custom_model_save_load(custom_model, custom_layer, data, custom_predicted, model_path):
    x, _ = data
    custom_objects = {"MyDense": custom_layer}
    mlflow.tensorflow.save_model(custom_model, path=model_path, custom_objects=custom_objects)
    # Loading Keras model
    model_loaded = mlflow.tensorflow.load_model(model_path)
    assert all(model_loaded.predict(x) == custom_predicted)
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    assert all(pyfunc_loaded.predict(x) == custom_predicted)

def test_model_log(model, data, predicted):
    x, _ = data
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "keras_model"
            model_info = mlflow.tensorflow.log_model(model, name=artifact_path)
            # Load model
            model_loaded = mlflow.tensorflow.load_model(model_uri=model_info.model_uri)
            assert all(model_loaded.predict(x) == predicted)
            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            assert all(pyfunc_loaded.predict(x) == predicted)
        finally:
            mlflow.end_run()

def test_log_model_with_pip_requirements(model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model, name="model", pip_requirements=str(req_file)
        )
        _assert_pip_requirements(model_info.model_uri, [expected_mlflow_version, "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model,
            name="model",
            pip_requirements=[f"-r {req_file}", "b"],
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model,
            name="model",
            pip_requirements=[f"-c {req_file}", "b"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )

def test_log_model_with_extra_pip_requirements(model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.tensorflow.get_default_pip_requirements()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model, name="model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model,
            name="model",
            extra_pip_requirements=[f"-r {req_file}", "b"],
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model,
            name="model",
            extra_pip_requirements=[f"-c {req_file}", "b"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )

def test_model_log_persists_requirements_in_mlflow_model_directory(model, keras_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model, name=artifact_path, conda_env=keras_custom_env
        )

    model_path = _download_artifact_from_uri(model_info.model_uri)
    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(keras_custom_env, saved_pip_req_path)

def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(model, keras_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model, name=artifact_path, conda_env=keras_custom_env
        )
        model_path = _download_artifact_from_uri(model_info.model_uri)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != keras_custom_env

    with open(keras_custom_env) as f:
        keras_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == keras_custom_env_parsed

def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(model):
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(model, name="model")
    _assert_pip_requirements(model_info.model_uri, mlflow.tensorflow.get_default_pip_requirements())

def test_model_load_succeeds_with_missing_data_key_when_data_exists_at_default_path(
    tf_keras_model, model_path, data
):
    """
    This is a backwards compatibility test to ensure that models saved in MLflow version <= 0.8.0
    can be loaded successfully. These models are missing the `data` flavor configuration key.
    """
    mlflow.tensorflow.save_model(
        tf_keras_model, path=model_path, keras_model_kwargs={"save_format": "h5"}
    )
    shutil.move(os.path.join(model_path, "data", "model.h5"), os.path.join(model_path, "model.h5"))
    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.tensorflow.FLAVOR_NAME, None)
    assert flavor_conf is not None
    del flavor_conf["data"]
    model_conf.save(model_conf_path)

    model_loaded = mlflow.tensorflow.load_model(model_path)
    assert all(model_loaded.predict(data[0]) == tf_keras_model.predict(data[0]))

def test_save_and_load_model_with_tf_save_format(tf_keras_model, model_path, data):
    mlflow.tensorflow.save_model(
        tf_keras_model, path=model_path, keras_model_kwargs={"save_format": "tf"}
    )
    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.tensorflow.FLAVOR_NAME, None)
    assert flavor_conf is not None
    assert flavor_conf.get("save_format") == "tf"
    assert not os.path.exists(os.path.join(model_path, "data", "model.h5")), (
        "TF model was saved with HDF5 format; expected SavedModel"
    )
    if Version(tf.__version__).release < (2, 16):
        assert os.path.isdir(os.path.join(model_path, "data", "model")), (
            "Expected directory containing saved_model.pb"
        )
    else:
        assert os.path.exists(os.path.join(model_path, "data", "model.keras")), (
            "Expected model saved as model.keras"
        )

    model_loaded = mlflow.tensorflow.load_model(model_path)
    np.testing.assert_allclose(model_loaded.predict(data[0]), tf_keras_model.predict(data[0]))

def test_load_without_save_format(tf_keras_model, model_path, data):
    mlflow.tensorflow.save_model(
        tf_keras_model, path=model_path, keras_model_kwargs={"save_format": "h5"}
    )
    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.tensorflow.FLAVOR_NAME)
    assert flavor_conf is not None
    del flavor_conf["save_format"]
    model_conf.save(model_conf_path)

    model_loaded = mlflow.tensorflow.load_model(model_path)
    np.testing.assert_allclose(model_loaded.predict(data[0]), tf_keras_model.predict(data[0]))

def test_pyfunc_serve_and_score_transformers():
    from transformers import BertConfig, TFBertModel

    bert_model = TFBertModel(
        BertConfig(
            vocab_size=16,
            hidden_size=2,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=2,
        )
    )
    dummy_inputs = bert_model.dummy_inputs["input_ids"].numpy()
    input_ids = tf.keras.layers.Input(shape=(dummy_inputs.shape[1],), dtype=tf.int32)
    model = tf.keras.Model(
        inputs=[input_ids], outputs=[bert_model.bert(input_ids).last_hidden_state]
    )
    model.compile()

    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            model,
            name="model",
            extra_pip_requirements=extra_pip_requirements,
            input_example=dummy_inputs,
        )

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )

    scores = PredictionsResponse.from_json(resp.content.decode("utf-8")).get_predictions(
        predictions_format="ndarray"
    )
    assert_array_almost_equal(scores, model.predict(dummy_inputs))

def test_model_log_with_metadata(tf_keras_model):
    artifact_path = "model"

    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            tf_keras_model, name=artifact_path, metadata={"metadata_key": "metadata_value"}
        )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"

def test_model_log_with_signature_inference(tf_keras_model, data, model_signature):
    artifact_path = "model"
    example = data[0][:3, :]

    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(
            tf_keras_model, name=artifact_path, input_example=example
        )

    mlflow_model = Model.load(model_info.model_uri)
    assert mlflow_model.signature == model_signature


# --- tests/tracing/utils/test_utils.py ---

def test_maybe_get_request_id():
    assert maybe_get_request_id(is_evaluate=True) is None

    try:
        from mlflow.pyfunc.context import Context, set_prediction_context
    except ImportError:
        pytest.skip("Skipping the rest of tests as mlflow.pyfunc module is not available.")

    with set_prediction_context(Context(request_id="eval", is_evaluate=True)):
        assert maybe_get_request_id(is_evaluate=True) == "eval"

    with set_prediction_context(Context(request_id="non_eval", is_evaluate=False)):
        assert maybe_get_request_id(is_evaluate=True) is None

def test_generate_trace_id_v4_from_otel_trace_id():
    otel_trace_id = 0x12345678901234567890123456789012
    location = "catalog.schema"

    result = generate_trace_id_v4_from_otel_trace_id(otel_trace_id, location)

    # Verify the format is trace:/<location>/<hex_trace_id>
    assert result.startswith(f"{TRACE_ID_V4_PREFIX}{location}/")

    # Extract and verify the hex trace ID part
    expected_hex_id = encode_trace_id(otel_trace_id)
    assert result == f"{TRACE_ID_V4_PREFIX}{location}/{expected_hex_id}"

    # Verify it can be parsed back
    parsed_location, parsed_id = parse_trace_id_v4(result)
    assert parsed_location == location
    assert parsed_id == expected_hex_id


# --- tests/tracking/fluent/test_fluent_autolog.py ---

def test_autolog_obeys_silent_mode(
    library,
    disable,
    exclusive,
    disable_for_unsupported_versions,
    log_models,
    log_datasets,
    log_input_examples,
    log_model_signatures,
):
    stream = StringIO()
    sys.stderr = stream

    mlflow.autolog(
        silent=True,
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        log_models=log_models,
        log_datasets=log_datasets,
        log_input_examples=log_input_examples,
        log_model_signatures=log_model_signatures,
    )

    mlflow.utils.import_hooks.notify_module_loaded(library)

    assert not stream.getvalue()

def test_last_active_run_retrieves_autologged_run():
    from sklearn.ensemble import RandomForestRegressor

    mlflow.autolog()
    rf = RandomForestRegressor(n_estimators=1, max_depth=1, max_features=1)
    rf.fit([[1, 2]], [[3]])
    rf.predict([[2, 1]])

    autolog_run = mlflow.last_active_run()
    assert autolog_run is not None
    assert autolog_run.info.run_id is not None

def test_extra_tags_mlflow_autolog():
    from sklearn.ensemble import RandomForestRegressor

    from mlflow.exceptions import MlflowException
    from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING

    mlflow.autolog(extra_tags={"test_tag": "autolog", MLFLOW_AUTOLOGGING: "123"})
    rf = RandomForestRegressor(n_estimators=1, max_depth=1, max_features=1)
    rf.fit([[1, 2]], [[3]])
    autolog_run = mlflow.last_active_run()
    assert autolog_run.data.tags["test_tag"] == "autolog"
    assert autolog_run.data.tags[MLFLOW_AUTOLOGGING] == "sklearn"

    with pytest.raises(MlflowException, match="Invalid `extra_tags` type"):
        mlflow.autolog(extra_tags="test_tag")


# --- tests/transformers/test_transformers_autolog.py ---

def test_setfit_does_not_autolog(setfit_trainer):
    mlflow.autolog()

    setfit_trainer.train()

    last_run = mlflow.last_active_run()
    assert not last_run
    preds = setfit_trainer.model([
        "Always carry a towel!",
        "The hobbits are going to Isengard",
        "What's tatoes, precious?",
    ])
    assert len(preds) == 3

def test_transformers_trainer_does_not_autolog_sklearn(transformers_trainer):
    mlflow.sklearn.autolog()

    exp = mlflow.set_experiment(experiment_name="trainer_autolog_test")

    transformers_trainer.train()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["_name_or_path"] == "distilbert-base-uncased"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    assert len(pipe("This is wonderful!")[0]["label"]) > 5  # Checking for 'LABEL_0' or 'LABEL_1'

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1

def test_transformers_autolog_adheres_to_global_behavior_using_setfit(setfit_trainer):
    mlflow.transformers.autolog(disable=False)

    setfit_trainer.train()
    assert len(mlflow.search_runs()) == 0
    preds = setfit_trainer.model(["Jim, I'm a doctor, not an archaeologist!"])
    assert len(preds) == 1

def test_transformers_autolog_adheres_to_global_behavior_using_trainer(transformers_trainer):
    mlflow.transformers.autolog()

    exp = mlflow.set_experiment(experiment_name="autolog_with_trainer")

    transformers_trainer.train()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    preds = pipe(["This is pretty ok, I guess", "I came here to chew bubblegum"])
    assert len(preds) == 2
    assert all(x["score"] > 0 for x in preds)

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1

def test_active_autolog_no_setfit_logging_followed_by_successful_sklearn_autolog(
    iris_data, setfit_trainer
):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="setfit_with_sklearn")

    # Train and evaluate
    setfit_trainer.train()
    metrics = setfit_trainer.evaluate()
    assert metrics["accuracy"] > 0

    # Run inference
    preds = setfit_trainer.model([
        "i loved the new Star Trek show!",
        "That burger was gross; it tasted like it was made from cat food!",
    ])
    assert len(preds) == 2

    # Test that autologging works for a simple sklearn model (local disabling functions)
    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    logged_sklearn_data = mlflow.get_run(run.info.run_id)
    assert logged_sklearn_data.data.tags["estimator_name"] == "KMeans"

    # Assert only the sklearn KMeans model was logged to the experiment

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    assert runs[0].info == logged_sklearn_data.info

def test_active_autolog_allows_subsequent_sklearn_autolog(iris_data, transformers_trainer):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="trainer_with_sklearn")

    transformers_trainer.train()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    preds = pipe(["This is pretty ok, I guess", "I came here to chew bubblegum"])
    assert len(preds) == 2
    assert all(x["score"] > 0 for x in preds)

    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    logged_sklearn_data = mlflow.get_run(run.info.run_id)
    assert logged_sklearn_data.data.tags["estimator_name"] == "KMeans"

    # Assert only the sklearn KMeans model was logged to the experiment

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 2
    sklearn_run = [x for x in runs if x.info.run_id == run.info.run_id]
    assert sklearn_run[0].info == logged_sklearn_data.info

def test_disabled_sklearn_autologging_does_not_revert_to_enabled_with_setfit(
    iris_data, setfit_trainer
):
    mlflow.autolog()
    mlflow.sklearn.autolog(disable=True)

    exp = mlflow.set_experiment(experiment_name="setfit_with_sklearn_no_autologging")

    # Train and evaluate
    setfit_trainer.train()
    metrics = setfit_trainer.evaluate()
    assert metrics["accuracy"] > 0

    # Run inference
    preds = setfit_trainer.model([
        "i loved the new Star Trek show!",
        "That burger was gross; it tasted like it was made from cat food!",
    ])
    assert len(preds) == 2

    # Test that autologging does not log since it is manually disabled above.
    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    # Assert that only the run info is logged
    logged_sklearn_data = mlflow.get_run(run.info.run_id)

    assert logged_sklearn_data.data.params == {}
    assert logged_sklearn_data.data.metrics == {}

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 1
    assert runs[0].info == logged_sklearn_data.info

def test_disable_sklearn_autologging_does_not_revert_with_trainer(iris_data, transformers_trainer):
    mlflow.autolog()
    mlflow.sklearn.autolog(disable=True)

    exp = mlflow.set_experiment(experiment_name="trainer_with_sklearn")

    transformers_trainer.train()
    mlflow.flush_async_logging()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    preds = pipe([
        "Did you hear that guitar solo? Brilliant!",
        "That band should avoid playing live.",
    ])
    assert len(preds) == 2
    assert all(x["score"] > 0 for x in preds)

    # Test that autologging does not log since it is manually disabled above.
    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    # Assert that only the run info is logged
    logged_sklearn_data = mlflow.get_run(run.info.run_id)

    assert logged_sklearn_data.data.params == {}
    assert logged_sklearn_data.data.metrics == {}

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 2
    sklearn_run = [x for x in runs if x.info.run_id == run.info.run_id]
    assert sklearn_run[0].info == logged_sklearn_data.info

def test_trainer_hyperparameter_tuning_does_not_log_sklearn_model(
    transformers_hyperparameter_trainer,
):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="hyperparam_trainer")

    transformers_hyperparameter_trainer.train()
    mlflow.flush_async_logging()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 3.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_hyperparameter_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    assert len(pipe("This is wonderful!")[0]["label"]) > 5  # checking for 'LABEL_0' or 'LABEL_1'

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 1

def test_trainer_hyperparameter_tuning_functional_does_not_log_sklearn_model(
    transformers_hyperparameter_functional,
):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="hyperparam_trainer_functional")

    transformers_hyperparameter_functional.train()
    mlflow.flush_async_logging()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_hyperparameter_functional.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    assert len(pipe("This is wonderful!")[0]["label"]) > 5  # checking for 'LABEL_0' or 'LABEL_1'

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 1


# --- tests/transformers/test_transformers_model_export.py ---

def test_inference_task_validation(small_qa_pipeline):
    with pytest.raises(
        MlflowException, match="The task provided is invalid. 'llm/v1/invalid' is not"
    ):
        _validate_llm_inference_task_type("llm/v1/invalid", "text-generation")
    with pytest.raises(
        MlflowException, match="The task provided is invalid. 'llm/v1/completions' is not"
    ):
        _validate_llm_inference_task_type("llm/v1/completions", small_qa_pipeline)
    _validate_llm_inference_task_type("llm/v1/completions", "text-generation")

def test_pipeline_construction_from_base_nlp_model(small_qa_pipeline):
    generated = _build_pipeline_from_model_input(
        {"model": small_qa_pipeline.model, "tokenizer": small_qa_pipeline.tokenizer},
        "question-answering",
    )
    assert isinstance(generated, type(small_qa_pipeline))
    assert isinstance(generated.tokenizer, type(small_qa_pipeline.tokenizer))

def test_pipeline_construction_from_base_vision_model(small_vision_model):
    model = {"model": small_vision_model.model, "tokenizer": small_vision_model.tokenizer}
    if IS_NEW_FEATURE_EXTRACTION_API:
        model.update({"image_processor": small_vision_model.image_processor})
    else:
        model.update({"feature_extractor": small_vision_model.feature_extractor})
    generated = _build_pipeline_from_model_input(model, task="image-classification")
    assert isinstance(generated, type(small_vision_model))
    assert isinstance(generated.tokenizer, type(small_vision_model.tokenizer))
    if IS_NEW_FEATURE_EXTRACTION_API:
        assert isinstance(generated.image_processor, type(small_vision_model.image_processor))
    else:
        assert isinstance(generated.feature_extractor, transformers.MobileNetV2ImageProcessor)

def test_log_and_load_transformers_pipeline(small_qa_pipeline, tmp_path, should_start_run):
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "transformers"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["transformers"])
        model_info = mlflow.transformers.log_model(
            small_qa_pipeline,
            name=artifact_path,
            conda_env=str(conda_env),
        )
        reloaded_model = mlflow.transformers.load_model(
            model_uri=model_info.model_uri, return_type="pipeline"
        )
        assert (
            reloaded_model(
                question="Who's house?", context="The house is owned by a man named Run."
            )["answer"]
            == "Run"
        )
        model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_info.model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
        assert model_path.joinpath(env_path).exists()
    finally:
        mlflow.end_run()

def test_transformers_log_model_with_prompt_template_sets_return_full_text_false(
    text_generation_pipeline,
):
    artifact_path = "text_generation_with_prompt_template"
    prompt_template = "User: {prompt}"

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            text_generation_pipeline,
            name=artifact_path,
            prompt_template=prompt_template,
        )

    model_path = pathlib.Path(_download_artifact_from_uri(model_info.model_uri))
    mlmodel = Model.load(str(model_path.joinpath("MLmodel")))

    pyfunc_flavor = mlmodel.flavors["python_function"]
    config = pyfunc_flavor.get("config")

    assert config.get("return_full_text") is False

def test_transformers_log_with_pip_requirements(small_multi_modal_pipeline, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()

    requirements_file = tmp_path.joinpath("requirements.txt")
    requirements_file.write_text("coolpackage")
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_multi_modal_pipeline, name="model", pip_requirements=str(requirements_file)
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, "coolpackage"], strict=True
        )
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            name="model",
            pip_requirements=[f"-r {requirements_file}", "alsocool"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "coolpackage", "alsocool"],
            strict=True,
        )
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            name="model",
            pip_requirements=[f"-c {requirements_file}", "constrainedcool"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "constrainedcool", "-c constraints.txt"],
            ["coolpackage"],
            strict=True,
        )

def test_transformers_log_with_extra_pip_requirements(small_multi_modal_pipeline, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_requirements = mlflow.transformers.get_default_pip_requirements(
        small_multi_modal_pipeline.model
    )
    requirements_file = tmp_path.joinpath("requirements.txt")
    requirements_file.write_text("coolpackage")
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_multi_modal_pipeline, name="model", extra_pip_requirements=str(requirements_file)
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_requirements, "coolpackage"],
            strict=True,
        )
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            name="model",
            extra_pip_requirements=[f"-r {requirements_file}", "alsocool"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_requirements, "coolpackage", "alsocool"],
            strict=True,
        )
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            name="model",
            extra_pip_requirements=[f"-c {requirements_file}", "constrainedcool"],
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [
                expected_mlflow_version,
                *default_requirements,
                "constrainedcool",
                "-c constraints.txt",
            ],
            ["coolpackage"],
            strict=True,
        )

def test_transformers_log_with_duplicate_extra_pip_requirements(small_multi_modal_pipeline):
    with pytest.raises(
        MlflowException, match="The specified requirements versions are incompatible"
    ):
        with mlflow.start_run():
            mlflow.transformers.log_model(
                small_multi_modal_pipeline,
                name="model",
                extra_pip_requirements=["transformers==1.1.0"],
            )

def test_transformers_pt_model_log_without_conda_env_uses_default_env_with_expected_dependencies(
    small_qa_pipeline,
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(small_qa_pipeline, name=artifact_path)
    _assert_pip_requirements(
        model_info.model_uri,
        mlflow.transformers.get_default_pip_requirements(small_qa_pipeline.model),
    )
    pip_requirements = _get_deps_from_requirement_file(model_info.model_uri)
    assert "tensorflow" not in pip_requirements
    assert "torch" in pip_requirements

def test_save_pipeline_without_defined_components(small_conversational_model, model_path):
    # This pipeline type explicitly does not have a configuration for an image_processor
    with mlflow.start_run():
        mlflow.transformers.save_model(
            transformers_model=small_conversational_model, path=model_path
        )
    pipe = mlflow.transformers.load_model(model_path)
    convo = transformers.Conversation("How are you today?")
    convo = pipe(convo)
    assert convo.generated_responses[-1] == "good"

def test_invalid_input_to_pyfunc_signature_output_wrapper_raises(component_multi_modal):
    with pytest.raises(MlflowException, match="The pipeline type submitted is not a valid"):
        mlflow.transformers.generate_signature_output(component_multi_modal["model"], "bogus")

def test_qa_pipeline_pyfunc_load_and_infer(small_qa_pipeline, model_path, inference_payload):
    signature = infer_signature(
        inference_payload,
        mlflow.transformers.generate_signature_output(small_qa_pipeline, inference_payload),
    )

    mlflow.transformers.save_model(
        transformers_model=small_qa_pipeline,
        path=model_path,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(inference_payload)

    assert isinstance(inference, list)
    assert all(isinstance(element, str) for element in inference)

    pd_input = (
        pd.DataFrame([inference_payload])
        if isinstance(inference_payload, dict)
        else pd.DataFrame(inference_payload)
    )
    pd_inference = pyfunc_loaded.predict(pd_input)

    assert isinstance(pd_inference, list)
    assert all(isinstance(element, str) for element in inference)

def test_vision_pipeline_pyfunc_load_and_infer(small_vision_model, model_path, inference_payload):
    if inference_payload == "base64":
        inference_payload = base64.b64encode(image_file_path.read_bytes()).decode("utf-8")
    elif inference_payload == "base64_encodebytes":
        inference_payload = base64.encodebytes(image_file_path.read_bytes()).decode("utf-8")
    signature = infer_signature(
        inference_payload,
        mlflow.transformers.generate_signature_output(small_vision_model, inference_payload),
    )
    mlflow.transformers.save_model(
        transformers_model=small_vision_model,
        path=model_path,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    predictions = pyfunc_loaded.predict(inference_payload)

    transformers_loaded_model = mlflow.transformers.load_model(model_path)
    expected_predictions = transformers_loaded_model.predict(inference_payload)
    assert list(predictions.to_dict("records")[0].values()) == expected_predictions

def test_text2text_generation_pipeline_with_model_configs(
    text2text_generation_pipeline, tmp_path, data, result
):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(text2text_generation_pipeline, data)
    )

    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    model_path1 = tmp_path.joinpath("model1")
    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path1,
        model_config=model_config,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path1)

    inference = pyfunc_loaded.predict(data)

    assert inference == result

    pd_input = pd.DataFrame([data]) if isinstance(data, str) else pd.DataFrame(data)
    pd_inference = pyfunc_loaded.predict(pd_input)
    assert pd_inference == result

    model_path2 = tmp_path.joinpath("model2")
    signature_with_params = infer_signature(
        data,
        mlflow.transformers.generate_signature_output(text2text_generation_pipeline, data),
        model_config,
    )
    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path2,
        signature=signature_with_params,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path2)

    dict_inference = pyfunc_loaded.predict(
        data,
        params=model_config,
    )

    assert dict_inference == inference

def test_text2text_generation_pipeline_with_model_config_and_params(
    text2text_generation_pipeline, model_path
):
    data = "muppet keyboard type"
    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
        "do_sample": True,
    }
    parameters = {"top_k": 3, "max_new_tokens": 30}
    generated_output = mlflow.transformers.generate_signature_output(
        text2text_generation_pipeline, data
    )
    signature = infer_signature(
        data,
        generated_output,
        parameters,
    )

    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path,
        model_config=model_config,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    # model_config and default params are all applied
    res = pyfunc_loaded.predict(data)
    applied_params = model_config.copy()
    applied_params.update(parameters)
    res2 = pyfunc_loaded.predict(data, applied_params)
    assert res == res2

    assert res != pyfunc_loaded.predict(data, {"max_new_tokens": 3})

    # Extra params are ignored
    assert res == pyfunc_loaded.predict(data, {"extra_param": "extra_value"})

def test_text2text_generation_pipeline_with_params_success(
    text2text_generation_pipeline, model_path
):
    data = "muppet keyboard type"
    parameters = {"top_k": 2, "num_beams": 5, "do_sample": True}
    generated_output = mlflow.transformers.generate_signature_output(
        text2text_generation_pipeline, data
    )
    signature = infer_signature(
        data,
        generated_output,
        parameters,
    )

    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    # parameters saved with ModelSignature is applied by default
    res = pyfunc_loaded.predict(data)
    res2 = pyfunc_loaded.predict(data, parameters)
    assert res == res2

def test_text2text_generation_pipeline_with_params_with_errors(
    text2text_generation_pipeline, model_path
):
    data = "muppet keyboard type"
    parameters = {"top_k": 2, "num_beams": 5, "invalid_param": "invalid_param", "do_sample": True}
    generated_output = mlflow.transformers.generate_signature_output(
        text2text_generation_pipeline, data
    )

    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path,
        signature=infer_signature(
            data,
            generated_output,
            parameters,
        ),
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    with pytest.raises(
        MlflowException,
        match=r"The params provided to the `predict` method are "
        r"not valid for pipeline Text2TextGenerationPipeline.",
    ):
        pyfunc_loaded.predict(data, parameters)

    # Type validation of params failure
    with pytest.raises(MlflowException, match=r"Invalid parameters found"):
        pyfunc_loaded.predict(data, {"top_k": "2"})

def test_text2text_generation_pipeline_with_inferred_schema(text2text_generation_pipeline):
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(text2text_generation_pipeline, name="my_model")
    pyfunc_loaded = mlflow.pyfunc.load_model(model_info.model_uri)

    assert pyfunc_loaded.predict("muppet board nails hammer")[0].startswith("A hammer")

def test_invalid_input_to_text2text_pipeline(text2text_generation_pipeline, invalid_data):
    # Adding this validation test due to the fact that we're constructing the input to the
    # Pipeline. The Pipeline requires a format of a pseudo-dict-like string. An example of
    # a valid input string: "answer: green. context: grass is primarily green in color."
    # We generate this string from a dict or generate a list of these strings from a list of
    # dictionaries.
    with pytest.raises(
        MlflowException, match=r"An invalid type has been supplied: .+\. Please supply"
    ):
        infer_signature(
            invalid_data,
            mlflow.transformers.generate_signature_output(
                text2text_generation_pipeline, invalid_data
            ),
        )

def test_text_generation_pipeline(text_generation_pipeline, model_path, data):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(text_generation_pipeline, data)
    )

    model_config = {
        "prefix": "software",
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    mlflow.transformers.save_model(
        text_generation_pipeline,
        path=model_path,
        model_config=model_config,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(data)

    if isinstance(data, list):
        assert inference[0].startswith(data[0])
        assert inference[1].startswith(data[1])
    else:
        assert inference[0].startswith(data)

    pd_input = pd.DataFrame([data], index=[0]) if isinstance(data, str) else pd.DataFrame(data)
    pd_inference = pyfunc_loaded.predict(pd_input)

    if isinstance(data, list):
        assert pd_inference[0].startswith(data[0])
        assert pd_inference[1].startswith(data[1])
    else:
        assert pd_inference[0].startswith(data)

def test_invalid_input_to_text_generation_pipeline(text_generation_pipeline, invalid_data):
    if isinstance(invalid_data, list):
        match = "If supplying a list, all values must be of string type"
    else:
        match = "The input data is of an incorrect type"
    with pytest.raises(MlflowException, match=match):
        infer_signature(
            invalid_data,
            mlflow.transformers.generate_signature_output(text_generation_pipeline, invalid_data),
        )

def test_fill_mask_pipeline(fill_mask_pipeline, model_path, inference_payload, result):
    signature = infer_signature(
        inference_payload,
        mlflow.transformers.generate_signature_output(fill_mask_pipeline, inference_payload),
    )

    mlflow.transformers.save_model(fill_mask_pipeline, path=model_path, signature=signature)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(inference_payload)
    assert inference == result

    if len(inference_payload) > 1 and isinstance(inference_payload, list):
        pd_input = pd.DataFrame([{"inputs": v} for v in inference_payload])
    elif isinstance(inference_payload, list) and len(inference_payload) == 1:
        pd_input = pd.DataFrame([{"inputs": v} for v in inference_payload], index=[0])
    else:
        pd_input = pd.DataFrame({"inputs": inference_payload}, index=[0])

    pd_inference = pyfunc_loaded.predict(pd_input)
    assert pd_inference == result

def test_fill_mask_pipeline_with_multiple_masks(fill_mask_pipeline, model_path):
    data = ["I <mask> the whole <mask> of <mask>", "I <mask> the whole <mask> of <mask>"]

    mlflow.transformers.save_model(fill_mask_pipeline, path=model_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(data)
    assert len(inference) == 2
    assert all(len(value) == 3 for value in inference)

def test_invalid_input_to_fill_mask_pipeline(fill_mask_pipeline, invalid_data):
    if isinstance(invalid_data, list):
        match = "Invalid data submission. Ensure all"
    else:
        match = "The input data is of an incorrect type"
    with pytest.raises(MlflowException, match=match):
        infer_signature(
            invalid_data,
            mlflow.transformers.generate_signature_output(fill_mask_pipeline, invalid_data),
        )

def test_zero_shot_classification_pipeline(zero_shot_pipeline, model_path, data):
    # NB: The list submission for this pipeline type can accept json-encoded lists or lists within
    # the values of the dictionary.
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(zero_shot_pipeline, data)
    )

    mlflow.transformers.save_model(zero_shot_pipeline, model_path, signature=signature)

    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    inference = loaded_pyfunc.predict(data)

    assert isinstance(inference, pd.DataFrame)
    if isinstance(data["sequences"], str):
        assert len(inference) == len(data["candidate_labels"])
    else:
        assert len(inference) == len(data["sequences"]) * len(data["candidate_labels"])

def test_table_question_answering_pipeline(table_question_answering_pipeline, model_path, query):
    table = {
        "Fruit": ["Apples", "Bananas", "Oranges", "Watermelon", "Blueberries"],
        "Sales": ["1230945.55", "86453.12", "11459.23", "8341.23", "2325.88"],
        "Inventory": ["910", "4589", "11200", "80", "3459"],
    }
    json_table = json.dumps(table)
    data = {"query": query, "table": json_table}
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(table_question_answering_pipeline, data)
    )

    mlflow.transformers.save_model(
        table_question_answering_pipeline, model_path, signature=signature
    )
    loaded = mlflow.pyfunc.load_model(model_path)

    inference = loaded.predict(data)
    assert len(inference) == 1 if isinstance(query, str) else len(query)

    pd_input = pd.DataFrame([data])
    pd_inference = loaded.predict(pd_input)
    assert pd_inference is not None

def test_custom_code_pipeline(custom_code_pipeline, model_path):
    data = "hello"

    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(custom_code_pipeline, data)
    )

    mlflow.transformers.save_model(
        custom_code_pipeline,
        path=model_path,
        signature=signature,
    )

    # just test that it doesn't blow up when performing inference
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    pyfunc_pred = pyfunc_loaded.predict(data)
    assert isinstance(pyfunc_pred[0][0], float)

    transformers_loaded = mlflow.transformers.load_model(model_path)
    transformers_pred = transformers_loaded(data)
    assert pyfunc_pred[0][0] == transformers_pred[0][0][0]

def test_custom_components_pipeline(custom_components_pipeline, model_path):
    data = "hello"

    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(custom_components_pipeline, data)
    )

    components = {
        "model": custom_components_pipeline.model,
        "tokenizer": custom_components_pipeline.tokenizer,
        "feature_extractor": custom_components_pipeline.feature_extractor,
    }

    mlflow.transformers.save_model(
        transformers_model=components, path=model_path, signature=signature
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    pyfunc_pred = pyfunc_loaded.predict(data)
    assert isinstance(pyfunc_pred[0][0], float)

    transformers_loaded = mlflow.transformers.load_model(model_path)
    transformers_pred = transformers_loaded(data)
    assert pyfunc_pred[0][0] == transformers_pred[0][0][0]

    # assert that all the reloaded components exist
    # and have the same class name as pre-save
    for name, component in components.items():
        assert component.__class__.__name__ == getattr(transformers_loaded, name).__class__.__name__

def test_translation_pipeline(translation_pipeline, model_path, data, result):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(translation_pipeline, data)
    )

    mlflow.transformers.save_model(translation_pipeline, path=model_path, signature=signature)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    inference = pyfunc_loaded.predict(data)
    assert inference == result

    if len(data) > 1 and isinstance(data, list):
        pd_input = pd.DataFrame([{"inputs": v} for v in data])
    elif isinstance(data, list) and len(data) == 1:
        pd_input = pd.DataFrame([{"inputs": v} for v in data], index=[0])
    else:
        pd_input = pd.DataFrame({"inputs": data}, index=[0])

    pd_inference = pyfunc_loaded.predict(pd_input)
    assert pd_inference == result

def test_summarization_pipeline(summarizer_pipeline, model_path, data):
    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "max_length": 90,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(summarizer_pipeline, data)
    )

    mlflow.transformers.save_model(
        summarizer_pipeline, path=model_path, model_config=model_config, signature=signature
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(data)
    if isinstance(data, list) and len(data) > 1:
        for i, entry in enumerate(data):
            assert inference[i].strip().startswith(entry)
    elif isinstance(data, list) and len(data) == 1:
        assert inference[0].strip().startswith(data[0])
    else:
        assert inference[0].strip().startswith(data)

    if len(data) > 1 and isinstance(data, list):
        pd_input = pd.DataFrame([{"inputs": v} for v in data])
    elif isinstance(data, list) and len(data) == 1:
        pd_input = pd.DataFrame([{"inputs": v} for v in data], index=[0])
    else:
        pd_input = pd.DataFrame({"inputs": data}, index=[0])

    pd_inference = pyfunc_loaded.predict(pd_input)
    if isinstance(data, list) and len(data) > 1:
        for i, entry in enumerate(data):
            assert pd_inference[i].strip().startswith(entry)
    elif isinstance(data, list) and len(data) == 1:
        assert pd_inference[0].strip().startswith(data[0])
    else:
        assert pd_inference[0].strip().startswith(data)

def test_classifier_pipeline(text_classification_pipeline, model_path, data):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(text_classification_pipeline, data)
    )
    mlflow.transformers.save_model(
        text_classification_pipeline, path=model_path, signature=signature
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    inference = pyfunc_loaded.predict(data)

    # verify that native transformers outputs match the pyfunc return values
    native_inference = text_classification_pipeline(data)
    inference_dict = inference.to_dict()

    if isinstance(data, str):
        assert len(inference) == 1
        assert inference_dict["label"][0] == native_inference[0]["label"]
        assert inference_dict["score"][0] == native_inference[0]["score"]
    else:
        assert len(inference) == len(data)
        for key in ["score", "label"]:
            for value in range(0, len(data)):
                if key == "label":
                    assert inference_dict[key][value] == native_inference[value][key]
                else:
                    assert math.isclose(
                        native_inference[value][key], inference_dict[key][value], rel_tol=1e-6
                    )

def test_conversational_pipeline(conversational_pipeline, model_path):
    assert mlflow.transformers._is_conversational_pipeline(conversational_pipeline)

    signature = infer_signature(
        "Hi there!",
        mlflow.transformers.generate_signature_output(conversational_pipeline, "Hi there!"),
    )

    mlflow.transformers.save_model(conversational_pipeline, model_path, signature=signature)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)

    first_response = loaded_pyfunc.predict("What is the best way to get to Antarctica?")

    assert first_response == "The best way would be to go to space."

    second_response = loaded_pyfunc.predict("What kind of boat should I use?")

    assert second_response == "The best way to get to space would be to reach out and touch it."

    # Test that a new loaded instance has no context.
    loaded_again_pyfunc = mlflow.pyfunc.load_model(model_path)
    third_response = loaded_again_pyfunc.predict("What kind of boat should I use?")

    assert third_response == "The one with the guns."

    fourth_response = loaded_again_pyfunc.predict("Can I use it to go to the moon?")

    assert fourth_response == "Sure."

def test_qa_pipeline_pyfunc_predict(small_qa_pipeline):
    artifact_path = "qa_model"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_qa_pipeline,
            name=artifact_path,
        )

    inference_payload = json.dumps({
        "inputs": {
            "question": [
                "What color is it?",
                "How do the people go?",
                "What does the 'wolf' howl at?",
            ],
            "context": [
                "Some people said it was green but I know that it's pink.",
                "The people on the bus go up and down. Up and down.",
                "The pack of 'wolves' stood on the cliff and a 'lone wolf' howled at "
                "the moon for hours.",
            ],
        }
    })
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [{0: "pink"}, {0: "up and down"}, {0: "the moon"}]

    inference_payload = json.dumps({
        "inputs": {
            "question": "Who's house?",
            "context": "The house is owned by a man named Run.",
        }
    })

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [{0: "Run"}]

def test_vision_pipeline_pyfunc_predict(small_vision_model, inference_payload):
    if inference_payload == "base64":
        inference_payload = [
            base64.b64encode(image_file_path.read_bytes()).decode("utf-8"),
        ]
    elif inference_payload == "base64_encodebytes":
        inference_payload = [
            base64.encodebytes(image_file_path.read_bytes()).decode("utf-8"),
        ]
    artifact_path = "image_classification_model"

    # Log the image classification model
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_vision_model,
            name=artifact_path,
        )
    pyfunc_inference_payload = json.dumps({"inputs": inference_payload})
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=pyfunc_inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    predictions = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    transformers_loaded_model = mlflow.transformers.load_model(model_info.model_uri)
    expected_predictions = transformers_loaded_model.predict(inference_payload)

    assert [list(pred.values()) for pred in predictions.to_dict("records")] == expected_predictions

def test_classifier_pipeline_pyfunc_predict(text_classification_pipeline):
    artifact_path = "text_classifier_model"
    data = [
        "I think this sushi might have gone off",
        "That gym smells like feet, hot garbage, and sadness",
        "I love that we have a moon",
        "I 'love' debugging subprocesses",
        'Quote "in" the string',
    ]
    signature = infer_signature(data)
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            text_classification_pipeline,
            name=artifact_path,
            signature=signature,
        )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": data}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 2
    assert len(values.to_dict()["score"]) == 5

    # test simple string input
    inference_payload = json.dumps({"inputs": ["testing"]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 2
    assert len(values.to_dict()["score"]) == 1

    # Test the alternate TextClassificationPipeline input structure where text_pair is used
    # and ensure that model serving and direct native inference match
    inference_data = [
        {"text": "test1", "text_pair": "pair1"},
        {"text": "test2", "text_pair": "pair2"},
        {"text": "test 'quote", "text_pair": "pair 'quote'"},
    ]
    signature = infer_signature(inference_data)
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            text_classification_pipeline,
            name=artifact_path,
            signature=signature,
        )

    inference_payload = json.dumps({"inputs": inference_data})
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    values_dict = values.to_dict()
    native_predict = text_classification_pipeline(inference_data)

    # validate that the pyfunc served model registers text_pair in the same manner as native
    for key in ["score", "label"]:
        for value in [0, 1]:
            if key == "label":
                assert values_dict[key][value] == native_predict[value][key]
            else:
                assert math.isclose(
                    values_dict[key][value], native_predict[value][key], rel_tol=1e-6
                )

def test_zero_shot_pipeline_pyfunc_predict(zero_shot_pipeline):
    artifact_path = "zero_shot_classifier_model"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            zero_shot_pipeline,
            name=artifact_path,
        )
        model_uri = model_info.model_uri

    inference_payload = json.dumps({
        "inputs": {
            "sequences": "My dog loves running through troughs of spaghetti with his mouth open",
            "candidate_labels": ["happy", "sad"],
            "hypothesis_template": "This example talks about how the dog is {}",
        }
    })

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 3
    assert len(values.to_dict()["labels"]) == 2

    inference_payload = json.dumps({
        "inputs": {
            "sequences": [
                "My dog loves to eat spaghetti",
                "My dog hates going to the vet",
                "My 'hamster' loves to play with my 'friendly' dog",
            ],
            "candidate_labels": '["happy", "sad"]',
            "hypothesis_template": "This example talks about how the dog is {}",
        }
    })
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 3
    assert len(values.to_dict()["labels"]) == 6

def test_table_question_answering_pyfunc_predict(table_question_answering_pipeline):
    artifact_path = "table_qa_model"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            table_question_answering_pipeline,
            name=artifact_path,
        )

    table = {
        "Fruit": ["Apples", "Bananas", "Oranges", "Watermelon 'small'", "Blueberries"],
        "Sales": ["1230945.55", "86453.12", "11459.23", "8341.23", "2325.88"],
        "Inventory": ["910", "4589", "11200", "80", "3459"],
    }

    inference_payload = json.dumps({
        "inputs": {
            "query": "What should we order more of?",
            "table": table,
        }
    })

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict(orient="records")) == 1

    inference_payload = json.dumps({
        "inputs": {
            "query": [
                "What is our highest sales?",
                "What should we order more of?",
                "Which 'fruit' has the 'highest' 'sales'?",
            ],
            "table": table,
        }
    })
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict(orient="records")) == 3

def test_feature_extraction_pipeline(feature_extraction_pipeline):
    sentences = ["hi", "hello"]
    signature = infer_signature(
        sentences,
        mlflow.transformers.generate_signature_output(feature_extraction_pipeline, sentences),
    )

    artifact_path = "feature_extraction_pipeline"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            feature_extraction_pipeline,
            name=artifact_path,
            signature=signature,
            input_example=["A sentence", "Another sentence"],
        )

    # Load as native
    loaded_pipeline = mlflow.transformers.load_model(model_info.model_uri)

    inference_single = "Testing"
    inference_mult = ["Testing something", "Testing something else"]

    pred = loaded_pipeline(inference_single)
    assert len(pred[0][0]) > 10
    assert isinstance(pred[0][0][0], float)

    pred_multiple = loaded_pipeline(inference_mult)
    assert len(pred_multiple[0][0]) > 2
    assert isinstance(pred_multiple[0][0][0][0], float)

    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

    pyfunc_pred = loaded_pyfunc.predict(inference_single)

    assert isinstance(pyfunc_pred, np.ndarray)

    assert np.array_equal(np.array(pred[0]), pyfunc_pred)

    pyfunc_pred_multiple = loaded_pyfunc.predict(inference_mult)

    assert np.array_equal(np.array(pred_multiple[0][0]), pyfunc_pred_multiple)

def test_feature_extraction_pipeline_pyfunc_predict(feature_extraction_pipeline):
    artifact_path = "feature_extraction"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            feature_extraction_pipeline,
            name=artifact_path,
        )

    inference_payload = json.dumps({"inputs": ["sentence one", "sentence two"]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.columns) == 384
    assert len(values) == 4

    inference_payload = json.dumps({"inputs": "sentence three"})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert response.status_code == 200
    prediction = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    assert len(prediction.columns) == 384
    assert len(prediction) == 4

def test_instructional_pipeline_no_prompt_in_output(model_path):
    architecture = "databricks/dolly-v2-3b"
    dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

    mlflow.transformers.save_model(
        transformers_model=dolly,
        path=model_path,
        # Validate removal of prompt but inclusion of newlines by default
        model_config={"max_length": 100, "include_prompt": False},
        input_example="Hello, Dolly!",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict("What is MLflow?")

    assert not inference[0].startswith("What is MLflow?")
    assert "\n" in inference[0]

def test_instructional_pipeline_no_prompt_in_output_and_removal_of_newlines(model_path):
    architecture = "databricks/dolly-v2-3b"
    dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

    mlflow.transformers.save_model(
        transformers_model=dolly,
        path=model_path,
        # Validate removal of prompt but inclusion of newlines by default
        model_config={"max_length": 100, "include_prompt": False, "collapse_whitespace": True},
        input_example="Hello, Dolly!",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict("What is MLflow?")

    assert not inference[0].startswith("What is MLflow?")
    assert "\n" not in inference[0]

def test_instructional_pipeline_with_prompt_in_output(model_path):
    architecture = "databricks/dolly-v2-3b"
    dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

    mlflow.transformers.save_model(
        transformers_model=dolly,
        path=model_path,
        # test default propagation of `include_prompt`=True and `collapse_whitespace`=False
        model_config={"max_length": 100},
        input_example="Hello, Dolly!",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict("What is MLflow?")

    assert inference[0].startswith("What is MLflow?")
    assert "\n\n" in inference[0]

def test_whisper_model_predict(model_path, whisper_pipeline, input_format, with_input_example):
    if input_format == "float" and not with_input_example:
        pytest.skip("If the input format is float, the default signature must be overridden.")

    audio = read_audio_data(input_format)
    mlflow.transformers.save_model(
        transformers_model=whisper_pipeline,
        path=model_path,
        input_example=audio if with_input_example else None,
        save_pretrained=False,
    )

    # 1. Single prediction with native transformer pipeline
    loaded_pipeline = mlflow.transformers.load_model(model_path)
    transcription = loaded_pipeline(audio)
    assert transcription["text"].startswith(" 30")
    # strip the leading space
    expected_text = transcription["text"].lstrip()

    # 2. Single prediction with Pyfunc
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    pyfunc_transcription = loaded_pyfunc.predict(audio)[0]
    assert pyfunc_transcription == expected_text

    # 3. Batch prediction with Pyfunc. Float input format is not supported for batch prediction,
    # because our signature framework doesn't support a list of numpy array.
    if input_format != "float":
        batch_transcription = loaded_pyfunc.predict([audio, audio])
        assert len(batch_transcription) == 2
        assert all(ts == expected_text for ts in batch_transcription)

def test_whisper_model_serve_and_score(whisper_pipeline):
    # Request payload to the model serving endpoint contains base64 encoded audio data
    audio = read_audio_data("bytes")
    encoded_audio = base64.b64encode(audio).decode("ascii")

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            whisper_pipeline,
            name="whisper",
            save_pretrained=False,
        )

    def _assert_response(response, length=1):
        preds = json.loads(response.content.decode("utf-8"))["predictions"]
        assert len(preds) == length
        assert all(pred.startswith("30") for pred in preds)

    with pyfunc_scoring_endpoint(
        model_info.model_uri,
        extra_args=["--env-manager", "local"],
    ) as endpoint:
        content_type = pyfunc_scoring_server.CONTENT_TYPE_JSON

        # Test payload with "inputs" key
        inputs_dict = {"inputs": [encoded_audio]}
        payload = json.dumps(inputs_dict)
        response = endpoint.invoke(payload, content_type=content_type)
        _assert_response(response)

        # Test payload with "dataframe_split" key
        inference_df = pd.DataFrame(pd.Series([encoded_audio], name="audio_file"))
        split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
        payload = json.dumps(split_dict)
        response = endpoint.invoke(payload, content_type=content_type)
        _assert_response(response)

        # Test payload with "dataframe_records" key
        records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
        payload = json.dumps(records_dict)
        response = endpoint.invoke(payload, content_type=content_type)
        _assert_response(response)

        # Test batch prediction
        inputs_dict = {"inputs": [encoded_audio, encoded_audio]}
        payload = json.dumps(inputs_dict)
        response = endpoint.invoke(payload, content_type=content_type)
        _assert_response(response, length=2)

        # Scoring with audio file URI is not supported yet (pyfunc prediction works tho)
        inputs_dict = {"inputs": [read_audio_data("file")]}
        payload = json.dumps(inputs_dict)
        response = endpoint.invoke(payload, content_type=content_type)
        response = json.loads(response.content.decode("utf-8"))
        assert response["error_code"] == "INVALID_PARAMETER_VALUE"
        assert "Failed to process the input audio data. Either" in response["message"]

def test_whisper_model_support_timestamps(whisper_pipeline):
    # Request payload to the model serving endpoint contains base64 encoded audio data
    audio = read_audio_data("bytes")
    encoded_audio = base64.b64encode(audio).decode("ascii")

    model_config = {
        "return_timestamps": "word",
        "chunk_length_s": 20,
        "stride_length_s": [5, 3],
    }

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            whisper_pipeline,
            name="whisper_timestamps",
            model_config=model_config,
            input_example=(audio, model_config),
        )

    # Native transformers prediction as ground truth
    gt = whisper_pipeline(audio, **model_config)

    def _assert_prediction(pred):
        assert pred["text"] == gt["text"]
        assert len(pred["chunks"]) == len(gt["chunks"])
        for pred_chunk, gt_chunk in zip(pred["chunks"], gt["chunks"]):
            assert pred_chunk["text"] == gt_chunk["text"]
            # Timestamps are tuples, but converted to list when serialized to JSON.
            assert tuple(pred_chunk["timestamp"]) == gt_chunk["timestamp"]

    # Prediction with Pyfunc
    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = json.loads(loaded_pyfunc.predict(audio)[0])
    _assert_prediction(prediction)

    # Serve and score
    with pyfunc_scoring_endpoint(
        model_info.model_uri,
        extra_args=["--env-manager", "local"],
    ) as endpoint:
        content_type = pyfunc_scoring_server.CONTENT_TYPE_JSON
        payload = json.dumps({"inputs": [encoded_audio]})
        response = endpoint.invoke(payload, content_type=content_type)

        predictions = json.loads(response.content.decode("utf-8"))["predictions"]
        # When return_timestamps is specified, the predictions list contains json
        # serialized output from the pipeline.
        _assert_prediction(json.loads(predictions[0]))

        # Request with inference params
        payload = json.dumps({
            "inputs": [encoded_audio],
            "model_config": model_config,
        })
        response = endpoint.invoke(payload, content_type=content_type)
        predictions = json.loads(response.content.decode("utf-8"))["predictions"]
        _assert_prediction(json.loads(predictions[0]))

def test_whisper_model_pyfunc_with_malformed_input(whisper_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=whisper_pipeline,
        path=model_path,
        save_pretrained=False,
    )

    pyfunc_model = mlflow.pyfunc.load_model(model_path)

    invalid_audio = b"This isn't a real audio file"
    with pytest.raises(MlflowException, match="Failed to process the input audio data. Either"):
        pyfunc_model.predict([invalid_audio])

    bad_uri_msg = "An invalid string input was provided. String"

    with pytest.raises(MlflowException, match=bad_uri_msg):
        pyfunc_model.predict("An invalid path")

    with pytest.raises(MlflowException, match=bad_uri_msg):
        pyfunc_model.predict("//www.invalid.net/audio.wav")

    with pytest.raises(MlflowException, match=bad_uri_msg):
        pyfunc_model.predict("https:///my/audio.mp3")

def test_audio_classification_pipeline(audio_classification_pipeline, with_input_example):
    audio = read_audio_data("bytes")

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            audio_classification_pipeline,
            name="audio_classification",
            input_example=audio if with_input_example else None,
            save_pretrained=False,
        )

    inference_payload = json.dumps({"inputs": [base64.b64encode(audio).decode("ascii")]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    assert isinstance(values, pd.DataFrame)
    assert len(values) > 1
    assert list(values.columns) == ["score", "label"]

def test_save_model_card_with_non_utf_characters(tmp_path, model_name):
    # non-ascii unicode characters
    test_text = (
        "Emoji testing! \u2728 \U0001f600 \U0001f609 \U0001f606 "
        "\U0001f970 \U0001f60e \U0001f917 \U0001f9d0"
    )

    card_data: ModelCard = huggingface_hub.ModelCard.load(model_name)
    card_data.text = card_data.text + "\n\n" + test_text
    custom_data = card_data.data.to_dict()
    custom_data["emojis"] = test_text

    card_data.data = huggingface_hub.CardData(**custom_data)
    _write_card_data(card_data, tmp_path)

    txt = tmp_path.joinpath(_CARD_TEXT_FILE_NAME).read_text()
    assert txt == card_data.text
    data = yaml.safe_load(tmp_path.joinpath(_CARD_DATA_FILE_NAME).read_text())
    assert data == card_data.data.to_dict()

def test_vision_pipeline_pyfunc_predict_with_kwargs(small_vision_model):
    artifact_path = "image_classification_model"

    parameters = {
        "top_k": 2,
    }
    inference_payload = json.dumps({
        "inputs": [image_url],
        "params": parameters,
    })

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_vision_model,
            name=artifact_path,
            signature=infer_signature(
                image_url,
                mlflow.transformers.generate_signature_output(small_vision_model, image_url),
                params=parameters,
            ),
        )
        model_uri = model_info.model_uri
    transformers_loaded_model = mlflow.transformers.load_model(model_uri)
    expected_predictions = transformers_loaded_model.predict(image_url)

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    predictions = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert (
        list(predictions.to_dict("records")[0].values())
        == expected_predictions[: parameters["top_k"]]
    )

def test_qa_pipeline_pyfunc_predict_with_kwargs(small_qa_pipeline):
    artifact_path = "qa_model"
    data = {
        "question": [
            "What color is it?",
            "What does the 'wolf' howl at?",
        ],
        "context": [
            "Some people said it was green but I know that it's pink.",
            "The pack of 'wolves' stood on the cliff and a 'lone wolf' howled at "
            "the moon for hours.",
        ],
    }
    parameters = {
        "top_k": 2,
        "max_answer_len": 5,
    }
    inference_payload = json.dumps({
        "inputs": data,
        "params": parameters,
    })
    output = mlflow.transformers.generate_signature_output(small_qa_pipeline, data)
    signature_with_params = infer_signature(
        data,
        output,
        parameters,
    )
    expected_signature = ModelSignature(
        Schema([
            ColSpec(Array(DataType.string), name="question"),
            ColSpec(Array(DataType.string), name="context"),
        ]),
        Schema([ColSpec(DataType.string)]),
        ParamSchema([
            ParamSpec("top_k", DataType.long, 2),
            ParamSpec("max_answer_len", DataType.long, 5),
        ]),
    )
    assert signature_with_params == expected_signature

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            small_qa_pipeline,
            name=artifact_path,
            signature=signature_with_params,
        )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [
        {0: "pink"},
        {0: "pink."},
        {0: "the moon"},
        {0: "moon"},
    ]

def test_uri_directory_renaming_handling_pipeline(model_path, text_classification_pipeline):
    with mlflow.start_run():
        mlflow.transformers.save_model(
            transformers_model=text_classification_pipeline, path=model_path
        )

    absolute_model_directory = os.path.join(model_path, "model")
    renamed_to_old_convention = os.path.join(model_path, "pipeline")
    os.rename(absolute_model_directory, renamed_to_old_convention)

    # remove the 'model_binary' entries to emulate older versions of MLflow
    mlmodel_file = os.path.join(model_path, "MLmodel")
    with open(mlmodel_file) as yaml_file:
        mlmodel = yaml.safe_load(yaml_file)

    mlmodel["flavors"]["python_function"].pop("model_binary", None)
    mlmodel["flavors"]["transformers"].pop("model_binary", None)

    with open(mlmodel_file, "w") as yaml_file:
        yaml.safe_dump(mlmodel, yaml_file)

    loaded_model = mlflow.pyfunc.load_model(model_path)

    prediction = loaded_model.predict("test")
    assert isinstance(prediction, pd.DataFrame)
    assert isinstance(prediction["label"][0], str)

def test_uri_directory_renaming_handling_components(model_path, text_classification_pipeline):
    components = {
        "tokenizer": text_classification_pipeline.tokenizer,
        "model": text_classification_pipeline.model,
    }

    with mlflow.start_run():
        mlflow.transformers.save_model(transformers_model=components, path=model_path)

    absolute_model_directory = os.path.join(model_path, "model")
    renamed_to_old_convention = os.path.join(model_path, "pipeline")
    os.rename(absolute_model_directory, renamed_to_old_convention)

    # remove the 'model_binary' entries to emulate older versions of MLflow
    mlmodel_file = os.path.join(model_path, "MLmodel")
    with open(mlmodel_file) as yaml_file:
        mlmodel = yaml.safe_load(yaml_file)

    mlmodel["flavors"]["python_function"].pop("model_binary", None)
    mlmodel["flavors"]["transformers"].pop("model_binary", None)

    with open(mlmodel_file, "w") as yaml_file:
        yaml.safe_dump(mlmodel, yaml_file)

    loaded_model = mlflow.pyfunc.load_model(model_path)

    prediction = loaded_model.predict("test")
    assert isinstance(prediction, pd.DataFrame)
    assert isinstance(prediction["label"][0], str)

def test_pyfunc_model_log_load_with_artifacts_snapshot():
    architecture = "prajjwal1/bert-tiny"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.BertForQuestionAnswering.from_pretrained(architecture)
    bert_tiny_pipeline = transformers.pipeline(
        task="question-answering", model=model, tokenizer=tokenizer
    )

    class QAModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            """
            This method initializes the tokenizer and language model
            using the specified snapshot location.
            """
            snapshot_location = context.artifacts["bert-tiny-model"].removeprefix("hf:/")
            # Initialize tokenizer and language model
            tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot_location)
            model = transformers.BertForQuestionAnswering.from_pretrained(snapshot_location)
            self.pipeline = transformers.pipeline(
                task="question-answering", model=model, tokenizer=tokenizer
            )

        def predict(self, context, model_input, params=None):
            question = model_input["question"][0]
            if isinstance(question, np.ndarray):
                question = question.item()
            ctx = model_input["context"][0]
            if isinstance(ctx, np.ndarray):
                ctx = ctx.item()
            return self.pipeline(question=question, context=ctx)

    data = {"question": "Who's house?", "context": "The house is owned by Run."}
    pyfunc_artifact_path = "question_answering_model"
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name=pyfunc_artifact_path,
            python_model=QAModel(),
            artifacts={"bert-tiny-model": "hf:/prajjwal1/bert-tiny"},
            input_example=data,
            signature=infer_signature(
                data, mlflow.transformers.generate_signature_output(bert_tiny_pipeline, data)
            ),
            extra_pip_requirements=["transformers", "torch", "numpy"],
        )

        pyfunc_model_path = _download_artifact_from_uri(model_info.model_uri)
        assert len(os.listdir(os.path.join(pyfunc_model_path, "artifacts"))) != 0
        model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert model_config.to_yaml() == loaded_pyfunc_model.metadata.to_yaml()
    assert loaded_pyfunc_model.predict(data)["answer"] != ""

    # Test model serving
    inference_payload = json.dumps({"inputs": data})
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records")[0]["answer"] != ""

def test_pyfunc_model_log_load_with_artifacts_snapshot_errors():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run():
        with pytest.raises(
            MlflowException,
            match=r"Failed to download snapshot from Hugging Face Hub "
            r"with artifact_uri: hf:/invalid-repo-id.",
        ):
            mlflow.pyfunc.log_model(
                name="pyfunc_artifact_path",
                python_model=TestModel(),
                artifacts={"some-model": "hf:/invalid-repo-id"},
            )

def test_text_generation_save_model_with_invalid_inference_task(
    text_generation_pipeline, model_path
):
    with pytest.raises(
        MlflowException, match=r"The task provided is invalid.*Must be.*llm/v1/completions"
    ):
        mlflow.transformers.save_model(
            transformers_model=text_generation_pipeline,
            path=model_path,
            task="llm/v1/invalid",
        )

def test_text_generation_log_model_with_mismatched_task(text_generation_pipeline):
    with pytest.raises(
        MlflowException, match=r"LLM v1 task type 'llm/v1/chat' is specified in metadata, but"
    ):
        with mlflow.start_run():
            mlflow.transformers.log_model(
                text_generation_pipeline,
                name="model",
                # Task argument and metadata task are different
                task=None,
                metadata={"task": "llm/v1/chat"},
            )

def test_text_generation_task_completions_predict_with_max_tokens(
    text_generation_pipeline, model_path
):
    mlflow.transformers.save_model(
        transformers_model=text_generation_pipeline,
        path=model_path,
        task="llm/v1/completions",
        model_config={"max_tokens": 10},
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(
        {"prompt": "How to learn Python in 3 weeks?", "max_tokens": 10},
    )

    assert isinstance(inference[0], dict)
    assert inference[0]["model"] == "distilgpt2"
    assert inference[0]["object"] == "text_completion"
    assert (
        inference[0]["choices"][0]["finish_reason"] == "length"
        and inference[0]["usage"]["completion_tokens"] == 10
    ) or (
        inference[0]["choices"][0]["finish_reason"] == "stop"
        and inference[0]["usage"]["completion_tokens"] < 10
    )

    # Override model_config with runtime params
    inference = pyfunc_loaded.predict(
        {"prompt": "How to learn Python in 3 weeks?", "max_tokens": 5},
    )
    assert 6 > inference[0]["usage"]["completion_tokens"] > 0

def test_text_generation_task_completions_predict_with_stop(text_generation_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=text_generation_pipeline,
        path=model_path,
        task="llm/v1/completions",
        metadata={"foo": "bar"},
        model_config={"stop": ["Python"], "max_tokens": 50},
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    inference = pyfunc_loaded.predict(
        {"prompt": "How to learn Python in 3 weeks?"},
    )

    if "Python" not in inference[0]["choices"][0]["text"]:
        pytest.skip(
            "Model did not generate text containing 'Python', "
            "skipping validation of stop parameter in inference"
        )

    assert (
        inference[0]["choices"][0]["finish_reason"] == "stop"
        and inference[0]["usage"]["completion_tokens"] < 50
    ) or (
        inference[0]["choices"][0]["finish_reason"] == "length"
        and inference[0]["usage"]["completion_tokens"] == 50
    )

    assert inference[0]["choices"][0]["text"].endswith("Python")

    # Override model_config with runtime params
    inference = pyfunc_loaded.predict(
        {"prompt": "How to learn Python in 3 weeks?", "stop": ["Abracadabra"]},
    )

    response_text = inference[0]["choices"][0]["text"]

    # Only check for early stopping if we stop on the word "Python".
    # If we exhaust the token limit, there is a non-insignificant chance of
    # terminating on the word due to max tokens, which should not count as
    # a stop word abort if there are multiple instances of the word in the text.
    if 0 < response_text.count("Python") < 2:
        assert not inference[0]["choices"][0]["text"].endswith("Python")

def test_text_generation_task_completions_serve(text_generation_pipeline):
    data = {"prompt": "How to learn Python in 3 weeks?"}

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            text_generation_pipeline,
            name="model",
            task="llm/v1/completions",
        )

    inference_payload = json.dumps({"inputs": data})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    output_dict = values.to_dict("records")[0]
    assert output_dict["choices"][0]["text"] is not None
    assert output_dict["choices"][0]["finish_reason"] == "stop"
    assert output_dict["usage"]["prompt_tokens"] < 20

def test_llm_v1_task_embeddings_predict(feature_extraction_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=feature_extraction_pipeline,
        path=model_path,
        input_examples=["Football", "Soccer"],
        task="llm/v1/embeddings",
    )

    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())

    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["inference_task"] == "llm/v1/embeddings"
    assert mlmodel["metadata"]["task"] == "llm/v1/embeddings"

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    # Predict with single string input
    prediction = pyfunc_loaded.predict({"input": "A great day"})
    assert prediction["object"] == "list"
    assert len(prediction["data"]) == 1
    assert prediction["data"][0]["object"] == "embedding"
    assert prediction["usage"]["prompt_tokens"] == 5
    assert len(prediction["data"][0]["embedding"]) == 384

    # Predict with list of string input
    prediction = pyfunc_loaded.predict({"input": ["A great day", "A bad day"]})
    assert prediction["object"] == "list"
    assert len(prediction["data"]) == 2
    assert prediction["data"][0]["object"] == "embedding"
    assert prediction["usage"]["prompt_tokens"] == 10
    assert len(prediction["data"][0]["embedding"]) == 384

    # Predict with pandas dataframe input
    df = pd.DataFrame({"input": ["A great day", "A bad day", "A good day"]})
    prediction = pyfunc_loaded.predict(df)
    assert prediction["object"] == "list"
    assert len(prediction["data"]) == 3
    assert prediction["data"][0]["object"] == "embedding"
    assert prediction["usage"]["prompt_tokens"] == 15
    assert len(prediction["data"][0]["embedding"]) == 384

def test_llm_v1_task_embeddings_serve(feature_extraction_pipeline, request_payload):
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            feature_extraction_pipeline,
            name="model",
            input_examples=["Football", "Soccer"],
            task="llm/v1/embeddings",
        )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps(request_payload),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    response = json.loads(response.content.decode("utf-8"))
    prediction = response["predictions"] if "inputs" in request_payload else response

    assert prediction["object"] == "list"
    assert len(prediction["data"]) == 1
    assert prediction["data"][0]["object"] == "embedding"
    assert len(prediction["data"][0]["embedding"]) == 384

def test_local_custom_model_save_and_load(text_generation_pipeline, model_path, tmp_path):
    local_repo_path = tmp_path / "local_repo"
    text_generation_pipeline.save_pretrained(local_repo_path)

    locally_loaded_model = transformers.AutoModelForCausalLM.from_pretrained(local_repo_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        local_repo_path, chat_template=CHAT_TEMPLATE
    )
    model_dict = {"model": locally_loaded_model, "tokenizer": tokenizer}

    # 1. Save local custom model without specifying task -> raises MlflowException
    with pytest.raises(MlflowException, match=r"The task could not be inferred"):
        mlflow.transformers.save_model(transformers_model=model_dict, path=model_path)

    # 2. Save local custom model with task -> saves successfully
    mlflow.transformers.save_model(
        transformers_model=model_dict,
        path=model_path,
        task="text-generation",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict("How to save Transformer model?")
    assert isinstance(inference[0], str)
    assert inference[0].startswith("How to save Transformer model?")

    # 3. Save local custom model with LLM v1 chat inference task -> saves successfully
    #    with the corresponding Transformers task
    shutil.rmtree(model_path)

    mlflow.transformers.save_model(
        transformers_model=model_dict,
        path=model_path,
        task="llm/v1/chat",
    )

    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["task"] == "text-generation"
    assert flavor_config["inference_task"] == "llm/v1/chat"
    assert mlmodel["metadata"]["task"] == "llm/v1/chat"

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict({
        "messages": [
            {
                "role": "user",
                "content": "How to save Transformer model?",
            }
        ]
    })
    assert isinstance(inference[0], dict)
    assert inference[0]["choices"][0]["message"]["role"] == "assistant"

def test_model_config_is_not_mutated_after_prediction(text2text_generation_pipeline):
    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "max_new_tokens": 500,
    }

    # Params will be used to override the values of model_config but should not mutate it
    params = {
        "top_k": 30,
        "max_length": 500,
        "max_new_tokens": 5,
    }

    pyfunc_model = _TransformersWrapper(text2text_generation_pipeline, model_config=model_config)
    assert pyfunc_model.model_config["top_k"] == 2

    prediction_output = pyfunc_model.predict(
        "rocket moon ship astronaut space gravity", params=params
    )

    assert pyfunc_model.model_config["top_k"] == 2
    assert pyfunc_model.model_config["num_beams"] == 5
    assert pyfunc_model.model_config["max_length"] == 30
    assert pyfunc_model.model_config["max_new_tokens"] == 500
    assert len(prediction_output[0].split(" ")) <= 5

def test_text_generation_task_chat_predict(text_generation_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=text_generation_pipeline,
        path=model_path,
        task="llm/v1/chat",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict({
        "messages": [
            {"role": "system", "content": "Hello, how can I help you today?"},
            {"role": "user", "content": "How to learn Python in 3 weeks?"},
        ],
        "max_tokens": 10,
    })

    assert inference[0]["choices"][0]["message"]["role"] == "assistant"
    assert (
        inference[0]["choices"][0]["finish_reason"] == "length"
        and inference[0]["usage"]["completion_tokens"] == 10
    ) or (
        inference[0]["choices"][0]["finish_reason"] == "stop"
        and inference[0]["usage"]["completion_tokens"] < 10
    )

def test_text_generation_task_chat_serve(text_generation_pipeline):
    data = {
        "messages": [
            {"role": "user", "content": "How to learn Python in 3 weeks?"},
        ],
        "max_tokens": 10,
    }

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            text_generation_pipeline,
            name="model",
            task="llm/v1/chat",
        )

    inference_payload = json.dumps(data)

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    output_dict = json.loads(response.content)[0]
    assert output_dict["choices"][0]["message"] is not None
    assert (
        output_dict["choices"][0]["finish_reason"] == "length"
        and output_dict["usage"]["completion_tokens"] == 10
    ) or (
        output_dict["choices"][0]["finish_reason"] == "stop"
        and output_dict["usage"]["completion_tokens"] < 10
    )
    assert output_dict["usage"]["prompt_tokens"] < 20

def test_save_model_from_local_checkpoint_with_llm_inference_task(
    model_path, local_checkpoint_path
):
    mlflow.transformers.save_model(
        transformers_model=local_checkpoint_path,
        path=model_path,
        task="llm/v1/chat",
        input_example=["What is MLflow?"],
    )

    logged_info = Model.load(model_path)
    flavor_conf = logged_info.flavors["transformers"]
    assert flavor_conf["source_model_name"] == local_checkpoint_path
    assert flavor_conf["task"] == "text-generation"
    assert flavor_conf["inference_task"] == "llm/v1/chat"

    # Load as pyfunc
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    response = loaded_pyfunc.predict({
        "messages": [
            {"role": "system", "content": "Hello, how can I help you today?"},
            {"role": "user", "content": "What is MLflow?"},
        ],
    })
    assert response[0]["choices"][0]["message"]["role"] == "assistant"
    assert response[0]["choices"][0]["message"]["content"] is not None


# --- tests/transformers/test_transformers_peft_model.py ---

def test_save_and_load_peft_pipeline(peft_pipeline, tmp_path):
    import peft

    from tests.transformers.test_transformers_model_export import HF_COMMIT_HASH_PATTERN

    mlflow.transformers.save_model(
        transformers_model=peft_pipeline,
        path=tmp_path,
    )

    # For PEFT, only the adapter model should be saved
    assert tmp_path.joinpath("peft").exists()
    assert not tmp_path.joinpath("model").exists()
    assert not tmp_path.joinpath("components").exists()

    # Validate the contents of MLModel file
    flavor_conf = Model.load(str(tmp_path.joinpath("MLmodel"))).flavors["transformers"]
    assert "model_binary" not in flavor_conf
    assert HF_COMMIT_HASH_PATTERN.match(flavor_conf["source_model_revision"])
    assert flavor_conf["peft_adaptor"] == "peft"

    # Validate peft is recorded to requirements.txt
    with open(tmp_path.joinpath("requirements.txt")) as f:
        assert f"peft=={peft.__version__}" in f.read()

    loaded_pipeline = mlflow.transformers.load_model(tmp_path)
    assert isinstance(loaded_pipeline.model, peft.PeftModel)
    loaded_pipeline.predict("Hi")

def test_save_and_load_peft_components(peft_pipeline, tmp_path, capsys):
    from peft import PeftModel

    mlflow.transformers.save_model(
        transformers_model={
            "model": peft_pipeline.model,
            "tokenizer": peft_pipeline.tokenizer,
        },
        path=tmp_path,
    )

    # PEFT pipeline construction error should not be raised
    peft_err_msg = (
        "The model 'PeftModelForSequenceClassification' is not supported for text-classification"
    )
    assert peft_err_msg not in capsys.readouterr().err

    loaded_pipeline = mlflow.transformers.load_model(tmp_path)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")

def test_log_peft_pipeline(peft_pipeline):
    from peft import PeftModel

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(peft_pipeline, name="model", input_example="hi")

    loaded_pipeline = mlflow.transformers.load_model(model_info.model_uri)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")

def test_save_and_load_peft_with_base_model_path(peft_model_with_local_base, tmp_path):
    from peft import PeftModel

    pipeline, base_dir = peft_model_with_local_base

    mlflow.transformers.save_model(
        transformers_model=pipeline,
        path=tmp_path,
        base_model_path=base_dir,
    )

    # PEFT adapter should be saved, components should be saved, but base model should NOT
    assert tmp_path.joinpath("peft").exists()
    assert not tmp_path.joinpath("model").exists()
    assert tmp_path.joinpath("components").exists()

    # Validate flavor config
    flavor_conf = Model.load(str(tmp_path.joinpath("MLmodel"))).flavors["transformers"]
    assert "model_binary" not in flavor_conf
    assert "source_model_revision" not in flavor_conf
    assert flavor_conf[FlavorKey.MODEL_LOCAL_BASE] == os.path.abspath(base_dir)
    assert flavor_conf[FlavorKey.PEFT] == "peft"

    loaded_pipeline = mlflow.transformers.load_model(tmp_path)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")

def test_log_peft_with_base_model_path(peft_model_with_local_base):
    from peft import PeftModel

    pipeline, base_dir = peft_model_with_local_base

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            pipeline,
            name="model",
            base_model_path=base_dir,
            input_example="hi",
        )

    loaded_pipeline = mlflow.transformers.load_model(model_info.model_uri)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")

def test_load_peft_with_base_model_path_override(peft_model_with_local_base, tmp_path):
    from peft import PeftModel

    pipeline, base_dir = peft_model_with_local_base
    save_dir = tmp_path / "model_output"

    # Save with a dummy path (simulating save on a different machine)
    mlflow.transformers.save_model(
        transformers_model=pipeline,
        path=save_dir,
        base_model_path=base_dir,
    )

    # Load with an explicit override path (simulating different mount point)
    loaded_pipeline = mlflow.transformers.load_model(save_dir, base_model_path=base_dir)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")

