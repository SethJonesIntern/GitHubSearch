# aws/sagemaker-python-sdk
# 45 LLM-backed test functions across 387 test files
# Source: https://github.com/aws/sagemaker-python-sdk

# --- sagemaker-mlops/tests/integ/test_clarify.py ---

def test_clarify_e2e(sagemaker_session, role, test_data, trained_model):
    model, X_test, y_test = trained_model
    bucket = sagemaker_session.default_bucket()
    prefix = 'clarify-test'
    data_filename = 'clarify_bias_test_data.csv'
    model_filename = 'clarify_test_model.joblib'
    
    # Prepare test data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(f'/tmp/{data_filename}', index=False)
    joblib.dump(model, f'/tmp/{model_filename}')
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(f'/tmp/{data_filename}', bucket, f'{prefix}/data/{data_filename}')
    s3_client.upload_file(f'/tmp/{model_filename}', bucket, f'{prefix}/model/{model_filename}')
    
    data_uri = f's3://{bucket}/{prefix}/data/{data_filename}'
    output_uri = f's3://{bucket}/{prefix}/output'
    
    # Configure Clarify
    data_config = DataConfig(
        s3_data_input_path=data_uri,
        s3_output_path=output_uri,
        label='target',
        headers=list(test_df.columns),
        dataset_type='text/csv'
    )
    
    bias_config = BiasConfig(
        label_values_or_threshold=[1],
        facet_name='gender',
        facet_values_or_threshold=[1]
    )
    
    shap_config = SHAPConfig(
        baseline=None,
        num_samples=10,
        agg_method='mean_abs'
    )
    
    # Create processor
    clarify_processor = SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        sagemaker_session=sagemaker_session
    )
    
    # Run pre-training bias analysis
    clarify_processor.run_pre_training_bias(
        data_config=data_config,
        data_bias_config=bias_config,
        methods=['CI', 'DPL'],
        wait=False,
        logs=False
    )
    
    assert clarify_processor.latest_job is not None
    job_name = clarify_processor.latest_job.get_name()
    
    try:
        # Poll for job completion
        timeout = 600  # 10 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = sagemaker_session.sagemaker_client.describe_processing_job(
                ProcessingJobName=job_name
            )
            status = response['ProcessingJobStatus']
            
            if status == 'Completed':
                assert status == 'Completed'
                break
            elif status in ['Failed', 'Stopped']:
                pytest.fail(f"Processing job {status}: {response.get('FailureReason', 'Unknown')}")
            
            time.sleep(30)  # Wait 1 minute
        else:
            pytest.fail(f"Processing job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()
        
        # Cleanup local files
        for f in [f'/tmp/{data_filename}', f'/tmp/{model_filename}']:
            if os.path.exists(f):
                os.remove(f)


# --- sagemaker-mlops/tests/integ/test_feature_store.py ---

def test_query_offline_store_with_athena(
    feature_group_name, sample_dataframe, bucket, role, sagemaker_session
):
    """Test querying offline store with Athena."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(300)
        
        # Note: Offline store sync can take 15+ minutes, test may return empty results
        athena_query = create_athena_query(feature_group_name, sagemaker_session)
        query_string = f'SELECT * FROM "{athena_query.database}"."{athena_query.table_name}" LIMIT 10'
        output_location = f"s3://{bucket}/athena-results/"
        
        query_id = athena_query.run(query_string, output_location)
        assert query_id is not None
        
        athena_query.wait()
        df = athena_query.as_dataframe()
        
        assert df is not None
        
    finally:
        cleanup_feature_group(feature_group_name)

def test_query_with_conditions_and_aggregations(
    feature_group_name, sample_dataframe, bucket, role, sagemaker_session
):
    """Test Athena queries with WHERE and aggregations."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(300)
        
        athena_query = create_athena_query(feature_group_name, sagemaker_session)
        query_string = f"""
            SELECT COUNT(*) as count, AVG(feature_1) as avg_feature
            FROM "{athena_query.database}"."{athena_query.table_name}"
            WHERE feature_2 > 5
        """
        output_location = f"s3://{bucket}/athena-results/"
        
        athena_query.run(query_string, output_location)
        athena_query.wait()
        df = athena_query.as_dataframe()
        
        assert df is not None
        
    finally:
        cleanup_feature_group(feature_group_name)


# --- sagemaker-mlops/tests/integ/test_feature_store_lakeformation.py ---

def test_create_feature_group_and_enable_lake_formation(s3_uri, role, region):
    """
    Test creating a FeatureGroupManager and enabling Lake Formation governance.

    This test:
    1. Creates a new FeatureGroupManager with offline store
    2. Waits for it to reach Created status
    3. Enables Lake Formation governance (registers S3, grants permissions, revokes IAM principals)
    4. Cleans up the FeatureGroupManager
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance
        result = fg.enable_lake_formation(hybrid_access_mode_enabled=False, acknowledge_risk=True)

        # Verify all phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_enabled"] is False

    finally:
        print('done')
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

def test_create_feature_group_with_lake_formation_enabled(s3_uri, role, region):
    """
    Test creating a FeatureGroupManager with lake_formation_config.enabled=True.

    This test verifies the integrated workflow where Lake Formation is enabled
    automatically during FeatureGroupManager creation:
    1. Creates a new FeatureGroupManager with lake_formation_config.enabled=True
    2. Verifies the FeatureGroupManager is created and Lake Formation is configured
    3. Cleans up the FeatureGroupManager
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager with Lake Formation enabled

        offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))
        lake_formation_config = LakeFormationConfig(
            enabled=True,
            hybrid_access_mode_enabled = False,
            acknowledge_risk=True,
        )

        fg = FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=role,
            lake_formation_config=lake_formation_config,
        )

        # Verify the FeatureGroupManager was created
        assert fg is not None
        assert fg.feature_group_name == fg_name
        assert fg.feature_group_status == "Created"

        # Verify Lake Formation is configured by checking we can refresh without errors
        fg.refresh()
        assert fg.offline_store_config is not None

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

def test_create_feature_group_without_lake_formation(s3_uri, role, region):
    """
    Test creating a FeatureGroupManager without Lake Formation enabled.

    This test verifies that when lake_formation_config is not provided or enabled=False,
    the FeatureGroupManager is created successfully without any Lake Formation operations:
    1. Creates a new FeatureGroupManager without lake_formation_config
    2. Verifies the FeatureGroupManager is created successfully
    3. Verifies no Lake Formation operations were performed
    4. Cleans up the FeatureGroupManager
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager without Lake Formation
        offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))

        # Create without lake_formation_config (default behavior)
        fg = FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=role,
        )

        # Verify the FeatureGroupManager was created
        assert fg is not None
        assert fg.feature_group_name == fg_name

        # Wait for Created status to ensure it's fully provisioned
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Verify offline store is configured
        fg.refresh()
        assert fg.offline_store_config is not None
        assert fg.offline_store_config.s3_storage_config is not None

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

def test_create_feature_group_with_lake_formation_fails_without_offline_store(role, region):
    """
    Test that creating a FeatureGroupManager with enable_lake_formation=True fails
    when no offline store is configured.

    Expected behavior: ValueError should be raised indicating offline store is required.
    """
    fg_name = generate_feature_group_name()

    lake_formation_config = LakeFormationConfig(hybrid_access_mode_enabled=False, acknowledge_risk=True)
    lake_formation_config.enabled = True

    # Attempt to create without offline store but with Lake Formation enabled
    with pytest.raises(ValueError) as exc_info:
        FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            lake_formation_config=lake_formation_config,
        )

    # Verify error message mentions offline_store_config requirement
    assert "lake_formation_config with enabled=True requires offline_store_config to be configured" in str(
        exc_info.value
    )

def test_create_feature_group_with_lake_formation_fails_without_role(s3_uri, region):
    """
    Test that creating a FeatureGroupManager with lake_formation_config.enabled=True fails
    when no role_arn is provided.

    Expected behavior: ValueError should be raised indicating role_arn is required.
    """
    fg_name = generate_feature_group_name()

    offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))
    lake_formation_config = LakeFormationConfig(hybrid_access_mode_enabled=False, acknowledge_risk=True)
    lake_formation_config.enabled = True

    # Attempt to create without role_arn but with Lake Formation enabled
    with pytest.raises(ValueError) as exc_info:
        FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            lake_formation_config=lake_formation_config,
        )

    # Verify error message mentions role_arn requirement
    assert "lake_formation_config with enabled=True requires role_arn to be specified" in str(exc_info.value)

def test_enable_lake_formation_fails_for_non_created_status(s3_uri, role, region):
    """
    Test that enable_lake_formation() fails when called on a FeatureGroupManager
    that is not in 'Created' status.

    Expected behavior: ValueError should be raised indicating the Feature Group
    must be in 'Created' status.

    Note: This test creates its own FeatureGroupManager because it needs to test
    behavior during the 'Creating' status, which requires a fresh resource.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Immediately try to enable Lake Formation without waiting for Created status
        # The Feature Group will be in 'Creating' status
        with pytest.raises(ValueError) as exc_info:
            fg.enable_lake_formation(hybrid_access_mode_enabled=False, acknowledge_risk=True, wait_for_active=False)

        # Verify error message mentions status requirement
        error_msg = str(exc_info.value)
        assert "must be in 'Created' status to enable Lake Formation" in error_msg

    finally:
        # Cleanup
        if fg:
            fg.wait_for_status(target_status="Created", poll=30, timeout=300)
            cleanup_feature_group(fg)

def test_enable_lake_formation_without_offline_store(role, region):
    """
    Test that enable_lake_formation() fails when called on a FeatureGroupManager
    without an offline store configured.

    Expected behavior: ValueError should be raised indicating offline store is required.

    Note: This test creates a FeatureGroupManager with only online store, which is a valid
    configuration, but Lake Formation cannot be enabled for it.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create a FeatureGroupManager with only online store (no offline store)
        online_store_config = OnlineStoreConfig(enable_online_store=True)

        fg = FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            online_store_config=online_store_config,
            role_arn=role,
        )

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)

        # Attempt to enable Lake Formation
        with pytest.raises(ValueError) as exc_info:
            fg.enable_lake_formation(hybrid_access_mode_enabled=False, acknowledge_risk=True)
        # Verify error message mentions offline store requirement
        assert "does not have an offline store configured" in str(exc_info.value)

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

def test_enable_lake_formation_fails_with_nonexistent_role(
    shared_feature_group_for_negative_tests, role
):
    """
    Test that enable_lake_formation() properly bubbles errors when using
    a nonexistent role ARN for Lake Formation registration.

    Expected behavior: RuntimeError or ClientError should be raised with details
    about the registration failure.

    Note: This test uses a nonexistent role ARN (current role with random suffix)
    to trigger an error during S3 registration with Lake Formation.
    """
    fg = shared_feature_group_for_negative_tests

    # Build a short nonexistent role ARN using the account ID from the real role
    account_id = role.split(":")[4]
    nonexistent_role = f"arn:aws:iam::{account_id}:role/non-existent-role"

    with pytest.raises(RuntimeError) as exc_info:
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=nonexistent_role,
            hybrid_access_mode_enabled=False,
            acknowledge_risk=True,
        )

    # Verify we got an appropriate error
    error_msg = str(exc_info.value)
    print(exc_info)
    # Should mention role-related issues (not found, invalid, access denied, etc.)
    assert "EntityNotFoundException" in error_msg

def test_enable_lake_formation_full_flow_with_policy_output(s3_uri, role, region, caplog):
    """
    Test the full Lake Formation flow with S3 deny policy logging.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with hybrid_access_mode_enabled=False
    3. Verifies all Lake Formation phases complete successfully
    4. Verifies the recommended S3 deny policy is logged as a warning
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance
        with caplog.at_level(logging.WARNING, logger="sagemaker.mlops.feature_store.feature_group_manager"):
            result = fg.enable_lake_formation(hybrid_access_mode_enabled=False, acknowledge_risk=True)

        # Verify all phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_enabled"] is False

        # Verify the recommended S3 deny policy was logged
        assert any("RECOMMENDED S3 BUCKET POLICY" in record.message for record in caplog.records)
        assert any("DenyFS" in record.message for record in caplog.records)

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

def test_enable_lake_formation_default_logs_recommended_policy(s3_uri, role, region, caplog):
    """
    Test that recommended bucket policy is logged with default arguments.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with hybrid_access_mode_enabled=False
    3. Verifies phases complete successfully (hybrid_access_mode_enabled=False)
    4. Verifies the recommended S3 deny policy is logged
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance with hybrid_access_mode_enabled=False
        with caplog.at_level(logging.WARNING, logger="sagemaker.mlops.feature_store.feature_group_manager"):
            result = fg.enable_lake_formation(hybrid_access_mode_enabled=False, acknowledge_risk=True)

        # Verify phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_enabled"] is False

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

def test_enable_lake_formation_with_custom_role_logs_policy(s3_uri, role, region, caplog):
    """
    Test the full Lake Formation flow with custom registration role.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with use_service_linked_role=False and a custom registration_role_arn
    3. Verifies all phases complete successfully
    4. Verifies the recommended S3 deny policy is logged
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation with custom registration role
        with caplog.at_level(logging.WARNING, logger="sagemaker.mlops.feature_store.feature_group_manager"):
            result = fg.enable_lake_formation(
                use_service_linked_role=False,
                registration_role_arn=role,
                hybrid_access_mode_enabled=False,
                acknowledge_risk=True,
            )

        # Verify all phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_enabled"] is False

        # Verify the recommended S3 deny policy was logged
        assert any("RECOMMENDED S3 BUCKET POLICY" in record.message for record in caplog.records)

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


# --- sagemaker-mlops/tests/integ/test_hyperparameter_tuning.py ---

def test_hyperparameter_tuning_e2e(sagemaker_session, role, mnist_data_dir):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "v3-tunning-integ-test"
    
    try:
        # Upload pre-downloaded MNIST data to S3
        s3_data_uri = sagemaker_session.upload_data(
            path=mnist_data_dir,
            bucket=bucket,
            key_prefix=f"{prefix}/data"
        )
        
        # Configure source code
        source_code = SourceCode(
            source_dir=os.path.join(os.path.dirname(__file__), "code"),
            entry_script="mnist.py"
        )
        
        # Configure compute
        compute = Compute(
            instance_type="ml.m5.xlarge",
            instance_count=1,
            volume_size_in_gb=30
        )
        
        # Configure stopping condition
        stopping_condition = StoppingCondition(
            max_runtime_in_seconds=3600
        )
        
        # Get training image
        training_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.10.0-gpu-py38"
        
        # Create ModelTrainer
        model_trainer = ModelTrainer(
            training_image=training_image,
            source_code=source_code,
            compute=compute,
            stopping_condition=stopping_condition,
            hyperparameters={
                "epochs": 1,
                "backend": "gloo"
            },
            sagemaker_session=sagemaker_session,
            role=role,
            base_job_name="test-hpo-pytorch"
        )
        
        # Define hyperparameter ranges
        hyperparameter_ranges = {
            "lr": ContinuousParameter(0.001, 0.1),
            "batch-size": CategoricalParameter([32, 64, 128]),
        }
        
        # Define metric definitions
        metric_definitions = [
            {
                "Name": "average test loss",
                "Regex": "Test set: Average loss: ([0-9\\.]+)"
            }
        ]
        
        # Create HyperparameterTuner
        tuner = HyperparameterTuner(
            model_trainer=model_trainer,
            objective_metric_name="average test loss",
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=metric_definitions,
            max_jobs=2,
            max_parallel_jobs=1,
            strategy="Random",
            objective_type="Minimize",
            early_stopping_type="Auto"
        )
        
        # Prepare input data
        training_data = InputData(
            channel_name="training",
            data_source=s3_data_uri
        )
        
        # Start tuning job
        tuner.tune(
            inputs=[training_data],
            wait=False
        )
        
        tuning_job_name = tuner._current_job_name
        assert tuning_job_name is not None
        
        # Poll for completion
        timeout = 1800  # 30 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = tuner.describe()
            status = response.hyper_parameter_tuning_job_status
            
            if status == "Completed":
                assert status == "Completed"
                break
            elif status in ["Failed", "Stopped"]:
                pytest.fail(f"Tuning job {status}")
            
            time.sleep(60)
        else:
            pytest.fail(f"Tuning job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()


# --- sagemaker-mlops/tests/integ/test_pipeline_train_registry.py ---

def test_pipeline_with_train_and_registry(sagemaker_session, pipeline_session, role):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "integ-test-v3-pipeline"
    base_job_prefix = "train-registry-job"

    # Upload abalone data to S3
    s3_client = boto3.client("s3")
    abalone_path = os.path.join(os.path.dirname(__file__), "data", "pipeline", "abalone.csv")
    s3_client.upload_file(abalone_path, bucket, f"{prefix}/input/abalone.csv")
    input_data_s3 = f"s3://{bucket}/{prefix}/input/abalone.csv"

    # Parameters
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=input_data_s3,
    )
    hyper_parameter_objective = ParameterString(
        name="TrainingObjective", default_value="reg:linear"
    )

    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    # Processing step
    sklearn_processor = ScriptProcessor(
        image_uri=image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            py_version="py3",
            instance_type="ml.m5.xlarge",
        ),
        instance_type=instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-sklearn",
        sagemaker_session=pipeline_session,
        role=role,
    )

    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                input_name="input-1",
                s3_input=ProcessingS3Input(
                    s3_uri=input_data,
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="ShardedByS3Key",
                ),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{sagemaker_session.default_bucket()}/{prefix}/train",
                    local_path="/opt/ml/processing/train",
                    s3_upload_mode="EndOfJob",
                ),
            ),
            ProcessingOutput(
                output_name="validation",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{sagemaker_session.default_bucket()}/{prefix}/validation",
                    local_path="/opt/ml/processing/validation",
                    s3_upload_mode="EndOfJob",
                ),
            ),
            ProcessingOutput(
                output_name="test",
                s3_output=ProcessingS3Output(
                    s3_uri=f"s3://{sagemaker_session.default_bucket()}/{prefix}/test",
                    local_path="/opt/ml/processing/test",
                    s3_upload_mode="EndOfJob",
                ),
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "code", "pipeline", "preprocess.py"),
        arguments=["--input-data", input_data],
    )

    step_process = ProcessingStep(
        name="PreprocessData",
        step_args=processor_args,
        cache_config=cache_config,
    )

    # Training step
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        compute=Compute(instance_type=instance_type, instance_count=training_instance_count),
        base_job_name=f"{base_job_prefix}-xgboost",
        sagemaker_session=pipeline_session,
        role=role,
        hyperparameters={
            "objective": hyper_parameter_objective,
            "num_round": 50,
            "max_depth": 5,
        },
        input_data_config=[
            InputData(
                channel_name="train",
                data_source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        ],
    )

    train_args = model_trainer.train()
    step_train = TrainingStep(
        name="TrainModel",
        step_args=train_args,
        cache_config=cache_config,
    )

    # Model step
    model_builder = ModelBuilder(
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=image_uri,
        sagemaker_session=pipeline_session,
        role_arn=role,
    )

    step_create_model = ModelStep(name="CreateModel", step_args=model_builder.build())

    # Register step
    model_package_group_name = f"integ-test-model-group-{uuid.uuid4().hex[:8]}"
    step_register_model = ModelStep(
        name="RegisterModel",
        step_args=model_builder.register(
            model_package_group_name=model_package_group_name,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            approval_status="Approved",
        ),
    )

    # Pipeline
    pipeline_name = f"integ-test-train-registry-{uuid.uuid4().hex[:8]}"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            training_instance_count,
            instance_type,
            input_data,
            hyper_parameter_objective,
        ],
        steps=[step_process, step_train, step_create_model, step_register_model],
        sagemaker_session=pipeline_session,
    )

    model_name = None
    try:
        # Upsert and execute pipeline
        pipeline.upsert(role_arn=role)
        execution = pipeline.start()

        # Poll execution status with 30 minute timeout
        timeout = 1800
        start_time = time.time()

        while time.time() - start_time < timeout:
            execution_desc = execution.describe()
            execution_status = execution_desc["PipelineExecutionStatus"]

            if execution_status == "Succeeded":
                # Get model name from execution steps
                steps = sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
                    PipelineExecutionArn=execution_desc["PipelineExecutionArn"]
                )["PipelineExecutionSteps"]
                for step in steps:
                    if step["StepName"] == "CreateModel" and "Metadata" in step:
                        model_name = step["Metadata"].get("Model", {}).get("Arn", "").split("/")[-1]
                        break
                assert execution_status == "Succeeded"
                break
            elif execution_status in ["Failed", "Stopped"]:
                # Get detailed failure information
                steps = sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
                    PipelineExecutionArn=execution_desc["PipelineExecutionArn"]
                )["PipelineExecutionSteps"]

                failed_steps = []
                for step in steps:
                    if step.get("StepStatus") == "Failed":
                        failure_reason = step.get("FailureReason", "Unknown reason")
                        failed_steps.append(f"{step['StepName']}: {failure_reason}")

                failure_details = (
                    "\n".join(failed_steps)
                    if failed_steps
                    else "No detailed failure information available"
                )
                pytest.fail(
                    f"Pipeline execution {execution_status}. Failed steps:\n{failure_details}"
                )

            time.sleep(60)
        else:
            pytest.fail(f"Pipeline execution timed out after {timeout} seconds")

    finally:
        # Cleanup S3 resources
        s3 = boto3.resource("s3")
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f"{prefix}/").delete()

        # Cleanup model
        if model_name:
            try:
                sagemaker_session.sagemaker_client.delete_model(ModelName=model_name)
            except Exception:
                pass

        # Cleanup model package group
        try:
            sagemaker_session.sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
        except Exception:
            pass

        # Cleanup pipeline
        try:
            sagemaker_session.sagemaker_client.delete_pipeline(PipelineName=pipeline_name)
        except Exception:
            pass


# --- sagemaker-mlops/tests/integ/test_processing_job_sklearn.py ---

def test_sklearn_processing_job(sagemaker_session, role, abalone_data_path):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = "integ-test-processing-sklearn"
    
    try:
        # Upload abalone data to S3
        input_s3_key = f"{prefix}/input/abalone.csv"
        s3_client = boto3.client('s3')
        s3_client.upload_file(abalone_data_path, bucket, input_s3_key)
        input_data = f"s3://{bucket}/{input_s3_key}"
        
        sklearn_processor = ScriptProcessor(
            image_uri=image_uris.retrieve(
                framework="sklearn",
                region=region,
                version="1.2-1",
                py_version="py3",
                instance_type="ml.m5.xlarge",
            ),
            instance_type="ml.m5.xlarge",
            instance_count=1,
            base_job_name="test-sklearn-preprocess",
            sagemaker_session=sagemaker_session,
            role=role,
        )
        
        processor_args = sklearn_processor.run(
            wait=False,
            inputs=[
                ProcessingInput(
                    input_name="input-1",
                    s3_input=ProcessingS3Input(
                        s3_uri=input_data,
                        local_path="/opt/ml/processing/input",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                        s3_data_distribution_type="ShardedByS3Key",
                    )
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/{prefix}/train",
                        local_path="/opt/ml/processing/train",
                        s3_upload_mode="EndOfJob"
                    )
                ),
                ProcessingOutput(
                    output_name="validation",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/{prefix}/validation",
                        local_path="/opt/ml/processing/validation",
                        s3_upload_mode="EndOfJob"
                    )
                ),
                ProcessingOutput(
                    output_name="test",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/{prefix}/test",
                        local_path="/opt/ml/processing/test",
                        s3_upload_mode="EndOfJob"
                    )
                ),
            ],
            code=os.path.join(os.path.dirname(__file__), "code", "preprocess.py"),
            arguments=["--input-data", input_data],
        )
        
        # Wait for processing job to complete
        timeout = 600  # 10 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            sklearn_processor.latest_job.refresh()
            status = sklearn_processor.latest_job.processing_job_status
            
            if status == "Completed":
                assert status == "Completed"
                break
            elif status in ["Failed", "Stopped"]:
                pytest.fail(f"Processing job {status}")
            
            time.sleep(30)
        else:
            pytest.fail(f"Processing job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()


# --- sagemaker-mlops/tests/integ/test_pytorch_processing.py ---

def test_pytorch_processing_job(sagemaker_session, role):
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    s3_prefix = "integ-test-pytorch-processing"
    processing_job_name = "{}-{}".format(s3_prefix, strftime("%d-%H-%M-%S", gmtime()))
    output_destination = "s3://{}/{}".format(bucket, s3_prefix)
    
    try:
        image_uri = get_training_image_uri(
            region=region,
            framework="pytorch",
            framework_version="1.13",
            py_version="py39",
            instance_type="ml.m5.xlarge",
        )
        
        pytorch_processor = FrameworkProcessor(
            image_uri=image_uri,
            role=role,
            instance_type="ml.m5.xlarge",
            instance_count=1,
        )
        
        pytorch_processor.run(
            code="preprocessing.py",
            source_dir=os.path.join(os.path.dirname(__file__), "code", "pytorch_processing"),
            requirements="requirements.txt",
            job_name=processing_job_name,
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri="{}/train".format(output_destination),
                        local_path="/opt/ml/processing/train",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
                ProcessingOutput(
                    output_name="test",
                    s3_output=ProcessingS3Output(
                        s3_uri="{}/test".format(output_destination),
                        local_path="/opt/ml/processing/test",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
            wait=False,
        )
        
        # Check job status with 10 minute timeout
        job = pytorch_processor.latest_job
        timeout = 600
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job.refresh()
            status = job.processing_job_status
            
            if status == "Completed":
                assert status == "Completed"
                break
            elif status in ["Failed", "Stopped"]:
                pytest.fail(f"Processing job {status}")
            
            time.sleep(30)
        else:
            pytest.fail(f"Processing job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{s3_prefix}/').delete()


# --- sagemaker-mlops/tests/integ/feature_store/feature_processor/test_feature_processor_integ.py ---

def test_feature_processor_transform_online_only_store_ingestion(
    sagemaker_session,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OnlineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        # this calls spark 3.3 which requires java 11
        transform() 

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 26

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        assert dataset.empty
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )

def test_feature_processor_transform_with_customized_data_source(
    sagemaker_session,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()

    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @attr.s
        class TestCSVDataSource(PySparkDataSource):

            s3_uri = attr.ib()
            data_source_name = "TestCSVDataSource"
            data_source_unique_id = "s3_uri"

            def read_data(self, spark, params) -> DataFrame:
                s3a_uri = self.s3_uri.replace("s3://", "s3a://")
                return spark.read.csv(s3a_uri, header=True, inferSchema=False)

        @feature_processor(
            inputs=[TestCSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OnlineStore"],
            spark_config={
                "spark.hadoop.fs.s3a.aws.credentials.provider": ",".join(
                    [
                        "com.amazonaws.auth.ContainerCredentialsProvider",
                        "com.amazonaws.auth.profile.ProfileCredentialsProvider",
                        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                    ]
                )
            },
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        transform()

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 26

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        assert dataset.empty
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )

def test_feature_processor_transform_offline_only_store_ingestion(
    sagemaker_session,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        transform()

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 0

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        dataset = dataset.drop(
            columns=["ingest_time", "write_time", "api_invocation_time", "is_deleted"]
        )

        expected = get_expected_dataframe()
        dataset_sorted = dataset.sort_values(by="id").reset_index(drop=True)
        expected_sorted = expected.sort_values(by="id").reset_index(drop=True)
        assert dataset_sorted.equals(expected_sorted)
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )

def test_feature_processor_transform_offline_only_store_ingestion_run_with_remote(
    sagemaker_session,
    pre_execution_commands,
    dependencies_path,
):
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @remote(
            pre_execution_commands=pre_execution_commands,
            dependencies=dependencies_path,
            spark_config=SparkConfig(),
            instance_type="ml.m5.xlarge",
        )
        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        transform()

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 0

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        dataset = dataset.drop(
            columns=["ingest_time", "write_time", "api_invocation_time", "is_deleted"]
        )

        expected = get_expected_dataframe()
        dataset_sorted = dataset.sort_values(by="id").reset_index(drop=True)
        expected_sorted = expected.sort_values(by="id").reset_index(drop=True)
        assert dataset_sorted.equals(expected_sorted)
    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )

def test_to_pipeline_and_execute(
    sagemaker_session,
    pre_execution_commands,
    dependencies_path,
):
    pipeline_name = "pipeline-name-01"
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @remote(
            pre_execution_commands=pre_execution_commands,
            dependencies=dependencies_path,
            spark_config=SparkConfig(),
            instance_type="ml.m5.xlarge",
        )
        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        _wait_for_feature_group_lineage_contexts(
            car_data_feature_group_name, sagemaker_session
        )

        pipeline_arn = to_pipeline(
            pipeline_name=pipeline_name,
            step=transform,
            role_arn=get_execution_role(sagemaker_session),
            max_retries=2,
            tags=[("integ_test_tag_key_1", "integ_test_tag_key_2")],
            sagemaker_session=sagemaker_session,
        )
        _sagemaker_client = get_sagemaker_client(sagemaker_session=sagemaker_session)

        assert pipeline_arn is not None

        tags = _sagemaker_client.list_tags(ResourceArn=pipeline_arn)["Tags"]

        tag_keys = [tag["Key"] for tag in tags]
        assert "integ_test_tag_key_1" in tag_keys

        pipeline_description = Pipeline(name=pipeline_name).describe()
        assert pipeline_arn == pipeline_description["PipelineArn"]
        assert get_execution_role(sagemaker_session) == pipeline_description["RoleArn"]

        pipeline_definition = json.loads(pipeline_description["PipelineDefinition"])
        assert len(pipeline_definition["Steps"]) == 1
        for retry_policy in pipeline_definition["Steps"][0]["RetryPolicies"]:
            assert retry_policy["MaxAttempts"] == 2

        pipeline_execution_arn = execute(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )

        status = _wait_for_pipeline_execution_to_reach_terminal_state(
            pipeline_execution_arn=pipeline_execution_arn,
            sagemaker_client=_sagemaker_client,
        )
        assert status == "Succeeded"

    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )

def test_schedule_and_event_trigger(
    sagemaker_session,
    pre_execution_commands,
    dependencies_path,
):
    pipeline_name = "pipeline-name-01"
    car_data_feature_group_name = get_car_data_feature_group_name()
    car_data_aggregated_feature_group_name = get_car_data_aggregated_feature_group_name()
    try:
        feature_groups = create_feature_groups(
            sagemaker_session=sagemaker_session,
            car_data_feature_group_name=car_data_feature_group_name,
            car_data_aggregated_feature_group_name=car_data_aggregated_feature_group_name,
            offline_store_s3_uri=get_offline_store_s3_uri(sagemaker_session=sagemaker_session),
        )

        raw_data_uri = get_raw_car_data_s3_uri(sagemaker_session=sagemaker_session)

        @remote(
            pre_execution_commands=pre_execution_commands,
            dependencies=dependencies_path,
            spark_config=SparkConfig(),
            instance_type="ml.m5.xlarge",
        )
        @feature_processor(
            inputs=[CSVDataSource(raw_data_uri)],
            output=feature_groups["car_data_arn"],
            target_stores=["OfflineStore"],
        )
        def transform(raw_s3_data_as_df):
            """Load data from S3, perform basic feature engineering, store it in a Feature Group"""
            from pyspark.sql.functions import regexp_replace
            from pyspark.sql.functions import lit

            transformed_df = (
                raw_s3_data_as_df
                # Rename Columns
                .withColumnRenamed("Id", "id")
                .withColumnRenamed("Model", "model")
                .withColumnRenamed("Year", "model_year")
                .withColumnRenamed("Status", "status")
                .withColumnRenamed("Mileage", "mileage")
                .withColumnRenamed("Price", "price")
                .withColumnRenamed("MSRP", "msrp")
                # Add Event Time
                .withColumn("ingest_time", lit(int(time.time())))
                # Remove punctuation and fluff; replace with NA
                .withColumn("Price", regexp_replace("Price", "\$", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "(,)|(mi\.)", ""))  # noqa: W605
                .withColumn("mileage", regexp_replace("mileage", "Not available", "NA"))
                .withColumn("price", regexp_replace("price", ",", ""))
                .withColumn("msrp", regexp_replace("msrp", "(^MSRP\s\\$)|(,)", ""))  # noqa: W605
                .withColumn("msrp", regexp_replace("msrp", "Not specified", "NA"))
                .withColumn("msrp", regexp_replace("msrp", "\\$\d+[a-zA-Z\s]+", "NA"))  # noqa: W605
                .withColumn("model", regexp_replace("model", "^\d\d\d\d\s", ""))  # noqa: W605
            )

            transformed_df.show()
            return transformed_df

        _wait_for_feature_group_lineage_contexts(
            car_data_feature_group_name, sagemaker_session
        )

        pipeline_arn = to_pipeline(
            pipeline_name=pipeline_name,
            step=transform,
            role_arn=get_execution_role(sagemaker_session),
            max_retries=2,
            sagemaker_session=sagemaker_session,
        )

        assert pipeline_arn is not None

        pipeline_description = Pipeline(name=pipeline_name).describe()
        assert pipeline_arn == pipeline_description["PipelineArn"]
        assert get_execution_role(sagemaker_session) == pipeline_description["RoleArn"]

        pipeline_definition = json.loads(pipeline_description["PipelineDefinition"])
        assert len(pipeline_definition["Steps"]) == 1
        for retry_policy in pipeline_definition["Steps"][0]["RetryPolicies"]:
            assert retry_policy["MaxAttempts"] == 2
        now = datetime.now(tz=pytz.utc)
        schedule_expression = f"at({now.strftime(SCHEDULE_EXPRESSION_TIMESTAMP_FORMAT)})"
        schedule(
            pipeline_name=pipeline_name,
            schedule_expression=schedule_expression,
            start_date=now,
            sagemaker_session=sagemaker_session,
        )
        time.sleep(60)
        executions = sagemaker_session.sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline_name
        )
        pipeline_execution_arn = executions["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]

        status = _wait_for_pipeline_execution_to_reach_terminal_state(
            pipeline_execution_arn=pipeline_execution_arn,
            sagemaker_client=get_sagemaker_client(sagemaker_session=sagemaker_session),
        )
        assert status == "Succeeded"

        featurestore_client = sagemaker_session.sagemaker_featurestore_runtime_client
        results = featurestore_client.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": car_data_feature_group_name,
                    "RecordIdentifiersValueAsString": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                    ],
                },
            ]
        )

        assert len(results["Records"]) == 0

        car_sales_query = create_athena_query(feature_group_name=car_data_feature_group_name, session=sagemaker_session)
        query = f'SELECT * FROM "sagemaker_featurestore".{car_sales_query.table_name} LIMIT 1000;'
        output_uri = "s3://{}/{}/input/data/{}".format(
            sagemaker_session.default_bucket(),
            "feature-processor-test",
            "csv-data-fg-result",
        )
        car_sales_query.run(query_string=query, output_location=output_uri)
        car_sales_query.wait()
        dataset = car_sales_query.as_dataframe()
        dataset = dataset.drop(
            columns=["ingest_time", "write_time", "api_invocation_time", "is_deleted"]
        )

        # assert dataset.equals(get_expected_dataframe())

        put_trigger(
            source_pipeline_events=[
                FeatureProcessorPipelineEvents(
                    pipeline_name=pipeline_name,
                    pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.FAILED],
                )
            ],
            target_pipeline=pipeline_name,
        )

        assert "trigger" in describe(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )
        assert describe(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)[
            "event_pattern"
        ] == json.dumps(
            {
                "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
                "source": ["aws.sagemaker"],
                "detail": {
                    "currentPipelineExecutionStatus": ["Failed"],
                    "pipelineArn": [pipeline_arn],
                },
            }
        )
        enable_trigger(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert (
            describe(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)[
                "trigger_state"
            ]
            == "ENABLED"
        )
        disable_trigger(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert (
            describe(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)[
                "trigger_state"
            ]
            == "DISABLED"
        )

        delete_schedule(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert "schedule_arn" not in describe(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )
        delete_trigger(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
        assert "trigger" not in describe(
            pipeline_name=pipeline_name, sagemaker_session=sagemaker_session
        )

    finally:
        cleanup_offline_store(
            feature_group=feature_groups["car_data_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_offline_store(
            feature_group=feature_groups["car_data_aggregated_feature_group"],
            sagemaker_session=sagemaker_session,
        )
        cleanup_feature_group(
            feature_groups["car_data_feature_group"], sagemaker_session=sagemaker_session
        )
        cleanup_feature_group(
            feature_groups["car_data_aggregated_feature_group"], sagemaker_session=sagemaker_session
        )


# --- sagemaker-mlops/tests/unit/sagemaker/mlops/feature_store/test_feature_group_manager.py ---

    def test_returns_list_not_dict(self):
        """Test that the method returns a list, not a dict."""
        result = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )
        assert isinstance(result, list)
        assert not isinstance(result, dict)

    def test_policy_includes_correct_bucket_arn_in_object_statement(self):
        """Test that the statements include correct bucket ARN and prefix in object actions statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        object_statement = statements[0]
        expected_resource = f"arn:aws:s3:::{bucket_name}/{s3_prefix}/*"
        assert object_statement["Resource"] == expected_resource

    def test_policy_includes_correct_bucket_arn_in_list_statement(self):
        """Test that the statements include correct bucket ARN in ListBucket statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        list_statement = statements[1]
        expected_resource = f"arn:aws:s3:::{bucket_name}"
        assert list_statement["Resource"] == expected_resource

    def test_policy_includes_correct_prefix_condition_in_list_statement(self):
        """Test that the statements include correct prefix condition in ListBucket statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        list_statement = statements[1]
        expected_prefix = f"{s3_prefix}/*"
        assert list_statement["Condition"]["StringLike"]["s3:prefix"] == expected_prefix

    def test_policy_preserves_bucket_name_exactly(self):
        """Test that bucket name is preserved exactly without modification."""
        test_cases = [
            "simple-bucket",
            "bucket.with.dots",
            "bucket-with-dashes-123",
            "mybucket",
            "a" * 63,
        ]

        for bucket_name in test_cases:
            statements = self.fg._generate_s3_deny_statements(
                bucket_name=bucket_name,
                s3_prefix="prefix",
                lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
                feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
            )

            assert bucket_name in statements[0]["Resource"]
            assert bucket_name in statements[1]["Resource"]

    def test_policy_preserves_prefix_exactly(self):
        """Test that S3 prefix is preserved exactly without modification."""
        test_cases = [
            "simple-prefix",
            "path/to/data",
            "feature-store/account-id/region/feature-group-name",
            "deep/nested/path/structure/data",
            "prefix_with_underscores",
            "prefix-with-dashes",
        ]

        for s3_prefix in test_cases:
            statements = self.fg._generate_s3_deny_statements(
                bucket_name="test-bucket",
                s3_prefix=s3_prefix,
                lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
                feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
            )

            assert f"{s3_prefix}/*" in statements[0]["Resource"]
            assert statements[1]["Condition"]["StringLike"]["s3:prefix"] == f"{s3_prefix}/*"

    def test_policy_has_correct_s3_arn_format(self):
        """Test that the statements use correct S3 ARN format (arn:aws:s3:::bucket/path)."""
        bucket_name = "test-bucket"
        s3_prefix = "test/prefix"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        object_resource = statements[0]["Resource"]
        assert object_resource.startswith("arn:aws:s3:::")
        assert object_resource == f"arn:aws:s3:::{bucket_name}/{s3_prefix}/*"

        list_resource = statements[1]["Resource"]
        assert list_resource.startswith("arn:aws:s3:::")
        assert list_resource == f"arn:aws:s3:::{bucket_name}"

    def test_policy_structure_validation(self):
        """Test that the statements have correct structure."""
        statements = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        assert isinstance(statements, list)
        assert len(statements) == 2

        object_statement = statements[0]
        assert object_statement["Sid"] == "DenyFSObjectAccess_prefix"
        assert object_statement["Effect"] == "Deny"
        assert object_statement["Principal"] == "*"
        assert "Condition" in object_statement
        assert "StringNotEquals" in object_statement["Condition"]

        list_statement = statements[1]
        assert list_statement["Sid"] == "DenyFSListAccess_prefix"
        assert list_statement["Effect"] == "Deny"
        assert list_statement["Principal"] == "*"
        assert "Condition" in list_statement
        assert "StringLike" in list_statement["Condition"]
        assert "StringNotEquals" in list_statement["Condition"]

    def test_policy_includes_both_principals_in_allowed_list(self):
        """Test that both Lake Formation role and Feature Store role are in allowed principals."""
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        object_principals = statements[0]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert lf_role_arn in object_principals
        assert fs_role_arn in object_principals
        assert len(object_principals) == 2

        list_principals = statements[1]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert lf_role_arn in list_principals
        assert fs_role_arn in list_principals
        assert len(list_principals) == 2

    def test_policy_has_correct_actions_in_each_statement(self):
        """Test that each statement has the correct S3 actions."""
        statements = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        object_actions = statements[0]["Action"]
        assert "s3:GetObject" in object_actions
        assert "s3:PutObject" in object_actions
        assert "s3:DeleteObject" in object_actions
        assert len(object_actions) == 3

        list_action = statements[1]["Action"]
        assert list_action == "s3:ListBucket"


# --- sagemaker-serve/tests/integ/test_optimize_integration.py ---

def test_optimize_build_deploy_invoke_cleanup():
    """Integration test for Optimize workflow"""
    logger.info("Starting Optimize integration test...")
    
    optimized_model = None
    core_endpoint = None
    
    try:
        # Build and deploy
        logger.info("Optimizing and deploying model...")
        optimized_model, core_endpoint = build_and_deploy()
        
        # Make prediction
        logger.info("Making prediction...")
        make_prediction(core_endpoint)
        
        # Test passed successfully
        logger.info("Optimize integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Optimize integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if optimized_model and core_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(optimized_model, core_endpoint)


# --- sagemaker-serve/tests/integ/test_tei_integration.py ---

def test_tei_build_deploy_invoke_cleanup():
    """Integration test for TEI model build, deploy, invoke, and cleanup workflow"""
    logger.info("Starting TEI integration test...")
    
    core_model = None
    core_endpoint = None
    
    try:
        # Build and deploy
        logger.info("Building and deploying TEI model...")
        core_model, core_endpoint = build_and_deploy()
        
        # Make prediction
        logger.info("Making prediction...")
        make_prediction(core_endpoint)
        
        # Test passed successfully
        logger.info("TEI integration test completed successfully")
        
    except Exception as e:
        logger.error(f"TEI integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if core_model and core_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(core_model, core_endpoint)


# --- sagemaker-serve/tests/integ/test_tgi_integration.py ---

def test_tgi_build_deploy_invoke_cleanup():
    """Integration test for TGI model build, deploy, invoke, and cleanup workflow"""
    logger.info("Starting TGI integration test...")
    
    core_model = None
    core_endpoint = None
    
    try:
        # Build and deploy
        logger.info("Building and deploying TGI model...")
        core_model, core_endpoint = build_and_deploy()
        
        # Make prediction
        logger.info("Making prediction...")
        make_prediction(core_endpoint)
        
        # Test passed successfully
        logger.info("TGI integration test completed successfully")
        
    except Exception as e:
        logger.error(f"TGI integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if core_model and core_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(core_model, core_endpoint)


# --- sagemaker-train/tests/integ/ai_registry/test_evaluator.py ---

    def test_create_reward_function_from_local_py_file_and_invoke(
        self, unique_name, sample_lambda_py_file, test_role, cleanup_list
    ):
        """End-to-end test: create evaluator from a raw .py file with non-default name and invoke it.

        Regression test for the handler name bug where the Lambda was created with an incorrect
        handler derived from the source filename instead of 'lambda_function.lambda_handler'.
        """
        import json
        import boto3

        evaluator = Evaluator.create(
            name=unique_name,
            type=REWARD_FUNCTION,
            source=sample_lambda_py_file,
            role=test_role,
            wait=True,  # wait for Lambda to be active
        )
        cleanup_list.append(evaluator)
        assert evaluator.method == EvaluatorMethod.BYOC
        assert evaluator.reference is not None

        # Wait for Lambda to become Active before invoking
        lambda_client = boto3.client("lambda")
        waiter = lambda_client.get_waiter("function_active_v2")
        waiter.wait(FunctionName=evaluator.reference)

        # Invoke the Lambda directly to verify the handler is correct
        lambda_client = boto3.client("lambda")
        response = lambda_client.invoke(
            FunctionName=evaluator.reference,
            InvocationType="RequestResponse",
            Payload=json.dumps({"input": "test"}).encode(),
        )
        assert response["StatusCode"] == 200
        assert "FunctionError" not in response, (
            f"Lambda invocation failed with error: {response.get('FunctionError')}"
        )
        result = json.loads(response["Payload"].read())
        assert result.get("statusCode") == 200


# --- sagemaker-train/tests/integ/train/test_rlaif_trainer_integration.py ---

def test_rlaif_trainer_lora_complete_workflow(sagemaker_session):
    """Test complete RLAIF training workflow with LORA."""
    
    rlaif_trainer = RLAIFTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        reward_model_id='openai.gpt-oss-120b-1:0',
        reward_prompt='Builtin.Summarize',
        mlflow_experiment_name="test-rlaif-finetuned-models-exp",
        mlflow_run_name="test-rlaif-finetuned-models-run",
        training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/rlvr-rlaif-oss-test-data/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )

    # Create training job
    training_job = rlaif_trainer.train(wait=False)
    
    # Manual wait loop to avoid resource_config issue
    max_wait_time = 3600  # 1 hour timeout
    poll_interval = 30    # Check every 30 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None

def test_rlaif_trainer_with_custom_reward_settings(sagemaker_session):
    """Test RLAIF trainer with different reward model and prompt."""

    rlaif_trainer = RLAIFTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        reward_model_id='openai.gpt-oss-120b-1:0',
        reward_prompt="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/JsonDoc/rlaif-test-prompt/0.0.1",
        mlflow_experiment_name="test-rlaif-finetuned-models-exp",
        mlflow_run_name="test-rlaif-finetuned-models-run",
        training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/rlvr-rlaif-oss-test-data/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )
    
    training_job = rlaif_trainer.train(wait=False)
    
    # Manual wait loop
    max_wait_time = 3600
    poll_interval = 30
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None

def test_rlaif_trainer_continued_finetuning(sagemaker_session):
    """Test complete RLAIF training workflow with LORA."""

    rlaif_trainer = RLAIFTrainer(
        model="arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        reward_model_id='openai.gpt-oss-120b-1:0',
        reward_prompt='Builtin.Summarize',
        mlflow_experiment_name="test-rlaif-finetuned-models-exp",
        mlflow_run_name="test-rlaif-finetuned-models-run",
        training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/rlvr-rlaif-oss-test-data/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )

    # Create training job
    training_job = rlaif_trainer.train(wait=False)

    # Manual wait loop to avoid resource_config issue
    max_wait_time = 3600  # 1 hour timeout
    poll_interval = 30  # Check every 30 seconds
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status

        if status in ["Completed", "Failed", "Stopped"]:
            break

        time.sleep(poll_interval)

    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None


# --- sagemaker-train/tests/integ/train/test_rlvr_trainer_integration.py ---

def test_rlvr_trainer_lora_complete_workflow(sagemaker_session):
    """Test complete RLVR training workflow with LORA."""
    
    rlvr_trainer = RLVRTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        mlflow_experiment_name="test-rlvr-finetuned-models-exp",
        mlflow_run_name="test-rlvr-finetuned-models-run",
        training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/rlvr-rlaif-oss-test-data/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        accept_eula=True
    )
    
    # Create training job
    training_job = rlvr_trainer.train(wait=False)
    
    # Manual wait loop to avoid resource_config issue
    max_wait_time = 3600  # 1 hour timeout
    poll_interval = 30    # Check every 30 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None

def test_rlvr_trainer_with_custom_reward_function(sagemaker_session):
    """Test RLVR trainer with custom reward function."""
    
    rlvr_trainer = RLVRTrainer(
        model="meta-textgeneration-llama-3-2-1b-instruct",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        mlflow_experiment_name="test-rlvr-finetuned-models-exp",
        mlflow_run_name="test-rlvr-finetuned-models-run",
        training_dataset="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/DataSet/rlvr-rlaif-oss-test-data/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing/output/",
        custom_reward_function="arn:aws:sagemaker:us-west-2:729646638167:hub-content/sdktest/JsonDoc/rlvr-test-rf/0.0.1",
        accept_eula=True
    )
    
    training_job = rlvr_trainer.train(wait=False)
    
    # Manual wait loop
    max_wait_time = 3600
    poll_interval = 30
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None

def test_rlvr_trainer_nova_workflow(sagemaker_session):
    """Test RLVR training workflow with Nova model."""
    import os
    os.environ['SAGEMAKER_REGION'] = 'us-east-1'

    # For fine-tuning 
    rlvr_trainer = RLVRTrainer(
        model="nova-textgeneration-lite-v2",
        model_package_group="sdk-test-finetuned-models",
        mlflow_experiment_name="test-nova-rlvr-finetuned-models-exp",
        mlflow_run_name="test-nova-rlvr-finetuned-models-run",
        training_dataset="s3://mc-flows-sdk-testing-us-east-1/input_data/rlvr-nova/grpo-64-sample.jsonl",
        validation_dataset="s3://mc-flows-sdk-testing-us-east-1/input_data/rlvr-nova/grpo-64-sample.jsonl",
        s3_output_path="s3://mc-flows-sdk-testing-us-east-1/output/",
        custom_reward_function="arn:aws:sagemaker:us-east-1:729646638167:hub-content/sdktest/JsonDoc/rlvr-nova-test-rf/0.0.1",
        accept_eula=True
    )
    rlvr_trainer.hyperparameters.data_s3_path = 's3://example-bucket'

    rlvr_trainer.hyperparameters.reward_lambda_arn = 'arn:aws:lambda:us-east-1:729646638167:function:rlvr-nova-reward-function'

    training_job = rlvr_trainer.train(wait=False)
    
    # Manual wait loop
    max_wait_time = 3600
    poll_interval = 30
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None


# --- sagemaker-train/tests/integ/train/test_sft_trainer_integration.py ---

def test_sft_trainer_nova_workflow(sagemaker_session):
    """Test SFT trainer with Nova model."""
    import os
    os.environ['SAGEMAKER_REGION'] = 'us-east-1'

    # For fine-tuning 
    sft_trainer_nova = SFTTrainer(
        model="nova-textgeneration-lite-v2",
        training_type=TrainingType.LORA, 
        model_package_group="sdk-test-finetuned-models",
        mlflow_experiment_name="test-nova-finetuned-models-exp",
        mlflow_run_name="test-nova-finetuned-models-run",
        training_dataset="arn:aws:sagemaker:us-east-1:729646638167:hub-content/sdktest/DataSet/sft-nova-test-dataset/0.0.1",
        s3_output_path="s3://mc-flows-sdk-testing-us-east-1/output/"
    )
    
    # Create training job
    training_job = sft_trainer_nova.train(wait=False)
    
    # Manual wait loop
    max_wait_time = 3600  # 1 hour timeout
    poll_interval = 30    # Check every 30 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status
        
        if status in ["Completed", "Failed", "Stopped"]:
            break
            
        time.sleep(poll_interval)
    
    # Verify job completed successfully
    assert training_job.training_job_status == "Completed"
    assert hasattr(training_job, 'output_model_package_arn')
    assert training_job.output_model_package_arn is not None

