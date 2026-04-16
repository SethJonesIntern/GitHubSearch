# aws/sagemaker-spark-container
# 8 LLM-backed test functions across 16 test files
# Source: https://github.com/aws/sagemaker-spark-container

# --- test/integration/history/test_spark_history_server.py ---

def test_history_server(tag, framework_version, role, image_uri, sagemaker_session, region, instance_type):
    print(
        f"PySparkProcessor args: tag={tag}, framework_version={framework_version}, "
        f"role={role}, image_uri={image_uri}, region={region}"
    )

    region = sagemaker_session.boto_region_name
    print(f"sagemaker_session region is: {region}")
    spark = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version=framework_version,
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )
    bucket = sagemaker_session.default_bucket()
    print(f"session bucket is: {bucket}")
    spark_event_logs_key_prefix = "spark/spark-history-fs"
    spark_event_logs_s3_uri = "s3://{}/{}".format(bucket, spark_event_logs_key_prefix)
    spark_event_log_local_path = "test/resources/data/files/sample_spark_event_logs"
    file_name = "sample_spark_event_logs"
    file_size = os.path.getsize(spark_event_log_local_path)

    with open("test/resources/data/files/sample_spark_event_logs") as data:
        body = data.read()
        S3Uploader.upload_string_as_file_body(
            body=body,
            desired_s3_uri=f"{spark_event_logs_s3_uri}/{file_name}",
            sagemaker_session=sagemaker_session,
        )

    _wait_for_file_to_be_uploaded(region, bucket, spark_event_logs_key_prefix, file_name, file_size)
    spark.start_history_server(spark_event_logs_s3_uri=spark_event_logs_s3_uri)

    try:
        response = _request_with_retry(HISTORY_SERVER_ENDPOINT)
        assert response.status == 200

        response = _request_with_retry(f"{HISTORY_SERVER_ENDPOINT}{SPARK_APPLICATION_URL_SUFFIX}", max_retries=15)
        print(f"Subpage response status code: {response.status}")
    finally:
        spark.terminate_history_server()

def test_history_server_with_expected_failure(
    tag, framework_version, role, image_uri, sagemaker_session, caplog, instance_type
):
    spark = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version=framework_version,
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )

    caplog.set_level(logging.ERROR)
    spark.start_history_server(spark_event_logs_s3_uri="invalids3uri")
    response = _request_with_retry(HISTORY_SERVER_ENDPOINT, max_retries=5)
    assert response is None
    assert "History server failed to start. Please run 'docker logs history_server' to see logs" in caplog.text


# --- test/integration/sagemaker/test_spark.py ---

def test_sagemaker_pyspark_multinode(
    role, image_uri, configuration, sagemaker_session, region, sagemaker_client, config, instance_type
):
    instance_count = config["instance_count"]
    python_version = config["py_version"]
    print(f"Creating job with {instance_count} instance count python version {python_version}")
    """Test that basic multinode case works on 32KB of data"""
    spark = PySparkProcessor(
        base_job_name="sm-spark-py",
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )
    bucket = spark.sagemaker_session.default_bucket()
    timestamp = datetime.now().isoformat()
    output_data_uri = "s3://{}/spark/output/sales/{}".format(bucket, timestamp)
    spark_event_logs_key_prefix = "spark/spark-events/{}".format(timestamp)
    spark_event_logs_s3_uri = "s3://{}/{}".format(bucket, spark_event_logs_key_prefix)

    with open("test/resources/data/files/data.jsonl") as data:
        body = data.read()
        input_data_uri = "s3://{}/spark/input/data.jsonl".format(bucket)
        S3Uploader.upload_string_as_file_body(
            body=body, desired_s3_uri=input_data_uri, sagemaker_session=sagemaker_session
        )

    script_name = "hello_py_spark_app_py39.py" if python_version == "py39" else "hello_py_spark_app.py"
    print(f"Running script {script_name}")
    spark.run(
        submit_app=f"test/resources/code/python/hello_py_spark/{script_name}",
        submit_py_files=["test/resources/code/python/hello_py_spark/hello_py_spark_udfs.py"],
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration=configuration,
        spark_event_logs_s3_uri=spark_event_logs_s3_uri,
        wait=False,
    )
    processing_job = spark.latest_job

    s3_client = boto3.client("s3", region_name=region)

    file_size = 0
    latest_file_size = None
    updated_times_count = 0
    time_out = time.time() + 900

    while not processing_job_not_fail_or_complete(sagemaker_client, processing_job.job_name):
        response = s3_client.list_objects(Bucket=bucket, Prefix=spark_event_logs_key_prefix)
        if "Contents" in response:
            # somehow when call list_objects the first file size is always 0, this for loop
            # is to skip that.
            for event_log_file in response["Contents"]:
                if event_log_file["Size"] != 0:
                    print("\n##### Latest file size is " + str(event_log_file["Size"]))
                    latest_file_size = event_log_file["Size"]

        # update the file size if it increased
        if latest_file_size and latest_file_size > file_size:
            print("\n##### S3 file updated.")
            updated_times_count += 1
            file_size = latest_file_size

        if time.time() > time_out:
            raise RuntimeError("Timeout")

        time.sleep(20)

    # verify that spark event logs are periodically written to s3
    print("\n##### file_size {} updated_times_count {}".format(file_size, updated_times_count))
    assert file_size != 0

    # Commenting this assert because it's flaky.
    # assert updated_times_count > 1

    output_contents = S3Downloader.list(output_data_uri, sagemaker_session=sagemaker_session)
    assert len(output_contents) != 0

def test_sagemaker_pyspark_sse_s3(role, image_uri, sagemaker_session, region, sagemaker_client, instance_type):
    """Test that Spark container can read and write S3 data encrypted with SSE-S3 (default AES256 encryption)"""
    spark = PySparkProcessor(
        base_job_name="sm-spark-py",
        image_uri=image_uri,
        role=role,
        instance_count=2,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )
    bucket = sagemaker_session.default_bucket()
    timestamp = datetime.now().isoformat()
    input_data_key = f"spark/input/sales/{timestamp}/data.jsonl"
    input_data_uri = f"s3://{bucket}/{input_data_key}"
    output_data_uri = f"s3://{bucket}/spark/output/sales/{timestamp}"
    s3_client = sagemaker_session.boto_session.client("s3", region_name=region)
    with open("test/resources/data/files/data.jsonl") as data:
        body = data.read()
        s3_client.put_object(Body=body, Bucket=bucket, Key=input_data_key, ServerSideEncryption="AES256")

    spark.run(
        submit_app="test/resources/code/python/hello_py_spark/hello_py_spark_app.py",
        submit_py_files=["test/resources/code/python/hello_py_spark/hello_py_spark_udfs.py"],
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration={
            "Classification": "core-site",
            "Properties": {"fs.s3a.server-side-encryption-algorithm": "AES256"},
        },
    )
    processing_job = spark.latest_job

    waiter = sagemaker_client.get_waiter("processing_job_completed_or_stopped")
    waiter.wait(
        ProcessingJobName=processing_job.job_name,
        # poll every 15 seconds. timeout after 15 minutes.
        WaiterConfig={"Delay": 15, "MaxAttempts": 60},
    )

    output_contents = S3Downloader.list(output_data_uri, sagemaker_session=sagemaker_session)
    assert len(output_contents) != 0

def test_sagemaker_pyspark_sse_kms_s3(
    role, image_uri, sagemaker_session, region, sagemaker_client, account_id, partition, instance_type
):
    spark = PySparkProcessor(
        base_job_name="sm-spark-py",
        image_uri=image_uri,
        role=role,
        instance_count=2,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )

    # This test expected AWS managed s3 kms key to be present. The key will be in
    # KMS > AWS managed keys > aws/s3
    kms_key_id = None
    kms_client = sagemaker_session.boto_session.client("kms", region_name=region)
    for alias in kms_client.list_aliases()["Aliases"]:
        if "s3" in alias["AliasName"]:
            kms_key_id = alias["TargetKeyId"]

    if not kms_key_id:
        raise ValueError("AWS managed s3 kms key(alias: aws/s3) does not exist")

    bucket = sagemaker_session.default_bucket()
    timestamp = datetime.now().isoformat()
    input_data_key = f"spark/input/sales/{timestamp}/data.jsonl"
    input_data_uri = f"s3://{bucket}/{input_data_key}"
    output_data_uri_prefix = f"spark/output/sales/{timestamp}"
    output_data_uri = f"s3://{bucket}/{output_data_uri_prefix}"
    s3_client = sagemaker_session.boto_session.client("s3", region_name=region)
    with open("test/resources/data/files/data.jsonl") as data:
        body = data.read()
        s3_client.put_object(
            Body=body, Bucket=bucket, Key=input_data_key, ServerSideEncryption="aws:kms", SSEKMSKeyId=kms_key_id
        )

    spark.run(
        submit_app="test/resources/code/python/hello_py_spark/hello_py_spark_app.py",
        submit_py_files=["test/resources/code/python/hello_py_spark/hello_py_spark_udfs.py"],
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration={
            "Classification": "core-site",
            "Properties": {
                "fs.s3a.server-side-encryption-algorithm": "SSE-KMS",
                "fs.s3a.server-side-encryption.key": f"arn:{partition}:kms:{region}:{account_id}:key/{kms_key_id}",
            },
        },
    )
    processing_job = spark.latest_job
    waiter = sagemaker_client.get_waiter("processing_job_completed_or_stopped")
    waiter.wait(
        ProcessingJobName=processing_job.job_name,
        # poll every 15 seconds. timeout after 15 minutes.
        WaiterConfig={"Delay": 15, "MaxAttempts": 60},
    )

    s3_objects = s3_client.list_objects(Bucket=bucket, Prefix=output_data_uri_prefix)["Contents"]
    assert len(s3_objects) != 0
    for s3_object in s3_objects:
        object_metadata = s3_client.get_object(Bucket=bucket, Key=s3_object["Key"])
        assert object_metadata["ServerSideEncryption"] == "aws:kms"
        assert object_metadata["SSEKMSKeyId"] == f"arn:{partition}:kms:{region}:{account_id}:key/{kms_key_id}"

def test_sagemaker_scala_jar_multinode(
    role, image_uri, configuration, sagemaker_session, sagemaker_client, instance_type
):
    """Test SparkJarProcessor using Scala application jar with external runtime dependency jars staged by SDK"""
    spark = SparkJarProcessor(
        base_job_name="sm-spark-scala",
        image_uri=image_uri,
        role=role,
        instance_count=2,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )

    bucket = spark.sagemaker_session.default_bucket()
    with open("test/resources/data/files/data.jsonl") as data:
        body = data.read()
        input_data_uri = "s3://{}/spark/input/data.jsonl".format(bucket)
        S3Uploader.upload_string_as_file_body(
            body=body, desired_s3_uri=input_data_uri, sagemaker_session=sagemaker_session
        )
    output_data_uri = "s3://{}/spark/output/sales/{}".format(bucket, datetime.now().isoformat())

    scala_project_dir = "test/resources/code/scala/hello-scala-spark"
    spark.run(
        submit_app="{}/target/scala-2.12/hello-scala-spark_2.12-1.0.jar".format(scala_project_dir),
        submit_class="com.amazonaws.sagemaker.spark.test.HelloScalaSparkApp",
        submit_jars=[
            "{}/lib_managed/jars/org.json4s/json4s-native_2.12/json4s-native_2.12-3.6.9.jar".format(scala_project_dir)
        ],
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration=configuration,
    )
    processing_job = spark.latest_job

    waiter = sagemaker_client.get_waiter("processing_job_completed_or_stopped")
    waiter.wait(
        ProcessingJobName=processing_job.job_name,
        # poll every 15 seconds. timeout after 15 minutes.
        WaiterConfig={"Delay": 15, "MaxAttempts": 60},
    )

    output_contents = S3Downloader.list(output_data_uri, sagemaker_session=sagemaker_session)
    assert len(output_contents) != 0

def test_sagemaker_feature_store_ingestion_multinode(
    sagemaker_session,
    sagemaker_client,
    spark_version,
    framework_version,
    image_uri,
    role,
    is_feature_store_available,
    instance_type,
):
    """Test FeatureStore use cases by ingesting data to feature group."""

    if not is_feature_store_available:
        pytest.skip("Skipping test due to feature store is not available in current region.")

    script_name = "py_spark_feature_store_ingestion.py"
    spark = PySparkProcessor(
        base_job_name="sm-spark-feature-store",
        image_uri=image_uri,
        role=role,
        instance_count=2,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )
    spark.run(
        submit_app=f"test/resources/code/python/feature_store_py_spark/{script_name}",
        wait=False,
    )

    processing_job = spark.latest_job
    waiter = sagemaker_client.get_waiter("processing_job_completed_or_stopped")

    waiter.wait(
        ProcessingJobName=processing_job.job_name,
        # poll every 15 seconds. timeout after 15 minutes.
        WaiterConfig={"Delay": 15, "MaxAttempts": 60},
    )

    response = sagemaker_client.describe_processing_job(ProcessingJobName=processing_job.job_name)

    if response["ProcessingJobStatus"] == "Stopped":
        raise RuntimeError("Feature store Spark job stopped unexpectedly")

def test_sagemaker_java_jar_multinode(
    tag, role, image_uri, configuration, sagemaker_session, sagemaker_client, instance_type
):
    """Test SparkJarProcessor using Java application jar"""
    spark = SparkJarProcessor(
        base_job_name="sm-spark-java",
        framework_version=tag,
        image_uri=image_uri,
        role=role,
        instance_count=2,
        instance_type=instance_type,
        max_runtime_in_seconds=1200,
        sagemaker_session=sagemaker_session,
    )

    bucket = spark.sagemaker_session.default_bucket()
    with open("test/resources/data/files/data.jsonl") as data:
        body = data.read()
        input_data_uri = "s3://{}/spark/input/data.jsonl".format(bucket)
        S3Uploader.upload_string_as_file_body(
            body=body, desired_s3_uri=input_data_uri, sagemaker_session=sagemaker_session
        )
    output_data_uri = "s3://{}/spark/output/sales/{}".format(bucket, datetime.now().isoformat())

    java_project_dir = "test/resources/code/java/hello-java-spark"
    spark.run(
        submit_app="{}/target/hello-java-spark-1.0-SNAPSHOT.jar".format(java_project_dir),
        submit_class="com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
        arguments=["--input", input_data_uri, "--output", output_data_uri],
        configuration=configuration,
    )
    processing_job = spark.latest_job

    waiter = sagemaker_client.get_waiter("processing_job_completed_or_stopped")
    waiter.wait(
        ProcessingJobName=processing_job.job_name,
        # poll every 15 seconds. timeout after 15 minutes.
        WaiterConfig={"Delay": 15, "MaxAttempts": 60},
    )

    output_contents = S3Downloader.list(output_data_uri, sagemaker_session=sagemaker_session)
    assert len(output_contents) != 0

