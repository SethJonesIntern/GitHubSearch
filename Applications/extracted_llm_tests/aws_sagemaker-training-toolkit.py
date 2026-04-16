# aws/sagemaker-training-toolkit
# 1 LLM-backed test functions across 26 test files
# Source: https://github.com/aws/sagemaker-training-toolkit

# --- test/functional/test_intermediate_output.py ---

def test_large_files():
    os.environ["TRAINING_JOB_NAME"] = _timestamp()
    p = intermediate_output.start_sync(bucket_uri, region)

    file_size = 1024 * 256 * 17  # 17MB

    file = os.path.join(intermediate_path, "file.npy")
    _generate_large_npy_file(file_size, file)

    file_to_modify = os.path.join(intermediate_path, "file_to_modify.npy")
    _generate_large_npy_file(file_size, file_to_modify)
    content_to_assert = _generate_large_npy_file(file_size, file_to_modify)

    files.write_failure_file("Failure!!")
    p.join()

    assert os.path.exists(file)
    assert os.path.exists(file_to_modify)

    key_prefix = os.path.join(os.environ.get("TRAINING_JOB_NAME"), "output", "intermediate")
    client = boto3.client("s3", region)
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file, intermediate_path))
    )
    assert _file_exists_in_s3(
        client, os.path.join(key_prefix, os.path.relpath(file_to_modify, intermediate_path))
    )

    # check that modified file has
    s3 = boto3.resource("s3", region_name=region)
    key = os.path.join(key_prefix, os.path.relpath(file_to_modify, intermediate_path))
    modified_file = os.path.join(environment.output_dir, "modified_file.npy")
    s3.Bucket(bucket).download_file(key, modified_file)
    assert np.array_equal(np.load(modified_file), content_to_assert)

