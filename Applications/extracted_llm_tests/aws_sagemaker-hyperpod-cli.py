# aws/sagemaker-hyperpod-cli
# 62 LLM-backed test functions across 85 test files
# Source: https://github.com/aws/sagemaker-hyperpod-cli

# --- test/integration_tests/cluster_management/test_cli_cluster_stack_deletion.py ---

def test_delete_with_user_confirmation(runner, cfn_client):
    """Test CLI deletion happy path with user confirmation."""
    # Create a test stack for this test
    import uuid
    stack_name = f"{TEST_STACK_PREFIX}-happy-{str(uuid.uuid4())[:8]}"
    create_test_stack(cfn_client, stack_name)
    
    try:
        # Test deletion with confirmation prompt (simulate 'y' response)
        result = runner.invoke(delete_cluster_stack, [
            stack_name,
            "--region", REGION
        ], input='y\n', catch_exceptions=False)
        
        assert_command_succeeded(result)
        assert_yes_no_prompt_displayed(result)
        assert_success_message_displayed(result, ["deletion", "initiated"])
        
        # Wait for deletion to complete
        wait_for_stack_delete_complete(cfn_client, stack_name)
        
        # Verify stack is deleted
        with pytest.raises(Exception) as exc_info:
            cfn_client.describe_stacks(StackName=stack_name)
        assert "does not exist" in str(exc_info.value)
        
    except Exception:
        # Cleanup in case of test failure
        try:
            cfn_client.delete_stack(StackName=stack_name)
        except:
            pass
        raise


# --- test/integration_tests/cluster_management/test_hp_cluster_creation.py ---

def test_init_cluster(runner, cluster_name):
    """Initialize cluster stack template and verify file creation."""
    result = runner.invoke(
        init, ["cluster-stack", "."], catch_exceptions=False
    )
    assert_command_succeeded(result)
    assert_init_files_created("./", "cluster-stack")

def test_validate_cluster(runner, cluster_name):
    """Validate cluster configuration for correctness."""
    result = runner.invoke(validate, catch_exceptions=False)
    assert_command_succeeded(result)

def test_create_cluster(runner, cluster_name, create_time):
    """Create cluster and verify submission messages."""
    global STACK_NAME, CREATE_TIME
    
    # Record time before submission
    CREATE_TIME = datetime.now(timezone.utc)
    
    result = runner.invoke(create, ["--region", REGION, "--template-version", "1"], catch_exceptions=False)
    assert_command_succeeded(result)
    
    # Verify expected submission messages appear
    assert "Submitted!" in result.output
    assert "Stack creation initiated" in result.output
    assert "Stack ID:" in result.output
    
    # Extract and store stack name for later tests with better error handling
    stack_id_match = re.search(r'Stack ID: (arn:aws:cloudformation[^\s]+)', result.output)
    if not stack_id_match:
        raise AssertionError(f"Stack ID not found in output: {result.output}")
    
    stack_id = stack_id_match.group(1)
    STACK_NAME = stack_id.split('/')[-2]
    
    print(f"✅ Successfully created stack: {STACK_NAME}")

def test_verify_cluster_submission_via_list(runner, cluster_name):
    """Use hyp list hyp-cluster to verify our stack was created and appears in the list."""
    global STACK_NAME, CREATE_TIME
    
    assert STACK_NAME, "Stack name should be set by previous test"
    assert CREATE_TIME, "Create time should be set by previous test"
    
    result = runner.invoke(list_cluster_stacks, ["--region", REGION], catch_exceptions=False)
    assert_command_succeeded(result)
    
    # Check that our stack appears in the list
    assert STACK_NAME in result.output, f"Stack {STACK_NAME} should appear in list output"
    
    # Check for recent creation times (within last 5 minutes of create)
    recent_threshold = CREATE_TIME - timedelta(minutes=1)
    creation_time_pattern = r'CreationTime\s+\|\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    creation_times = re.findall(creation_time_pattern, result.output)
    
    recent_creations = []
    for time_str in creation_times:
        try:
            # Use fromisoformat for better performance with ISO dates
            iso_time_str = time_str.replace(' ', 'T')
            creation_time = datetime.fromisoformat(iso_time_str).replace(tzinfo=timezone.utc)
            if creation_time >= recent_threshold:
                recent_creations.append(creation_time)
        except ValueError:
            # Fallback to strptime for non-ISO format
            try:
                creation_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                if creation_time >= recent_threshold:
                    recent_creations.append(creation_time)
            except ValueError:
                continue
    
    assert recent_creations, f"Should have recent stack creations after {CREATE_TIME}"
    print(f"✅ Found {len(recent_creations)} recent stack creations, including our created stack")

def test_describe_cluster_via_cli(runner, cluster_name):
    """Use hyp describe to get details about our created stack."""
    global STACK_NAME
    
    assert STACK_NAME, "Stack name should be set by previous test"
    
    # Try to describe the stack using CLI
    result = runner.invoke(describe_cluster_stack, [STACK_NAME, "--region", REGION], catch_exceptions=False)
    
    assert_command_succeeded(result)
    assert STACK_NAME in result.output, f"Stack {STACK_NAME} should appear in describe output"
    assert "StackStatus" in result.output or "Status" in result.output, "Stack status should be shown"

def test_wait_for_stack_completion(runner, cluster_name):
    """Wait for CloudFormation stack to be fully complete."""
    global STACK_NAME
    assert STACK_NAME, "Stack name should be available"
    
    print(f"⏳ Waiting for CloudFormation stack {STACK_NAME} to be CREATE_COMPLETE...")
    wait_for_stack_complete(STACK_NAME, REGION)
    print(f"✅ Stack {STACK_NAME} is now CREATE_COMPLETE")

def test_cluster_update_workflow(runner, cluster_name):
    """Test hyp update-cluster command by toggling node recovery setting."""
    global STACK_NAME
    
    # Get initial node recovery setting
    initial_recovery = get_node_recovery_setting(cluster_name, REGION)
    print(f"Initial NodeRecovery setting: {initial_recovery}")
    
    # Determine target setting (toggle to opposite)
    target_recovery = "None" if initial_recovery == "Automatic" else "Automatic"
    print(f"Will change NodeRecovery to: {target_recovery}")
    
    # Test hyp update command
    result = runner.invoke(update_cluster, [
        "--cluster-name", cluster_name,
        "--node-recovery", target_recovery,
        "--region", REGION
    ], catch_exceptions=False)
    
    assert_command_succeeded(result)
    assert f"Cluster {cluster_name} has been updated" in result.output
    
    print(f"✅ Successfully ran hyp update-cluster command")

    # Get the current setting after update
    current_recovery = get_node_recovery_setting(cluster_name, REGION)
    print(f"Current NodeRecovery setting after update: {current_recovery}")
    
    # Verify the setting is valid and has been updated
    assert current_recovery in ["Automatic", "None"], f"Invalid NodeRecovery value: {current_recovery}"
    assert current_recovery != initial_recovery, f"NodeRecovery should have changed from {initial_recovery}"
    
    print(f"✅ Cluster update verification successful - NodeRecovery is now {current_recovery}")


# --- test/integration_tests/cluster_management/test_sdk_cluster_stack_deletion.py ---

def test_sdk_delete_basic_functionality(cfn_client):
    """Test basic SDK deletion functionality with auto-confirmation."""
    # Create test stack
    stack_name = f"{TEST_STACK_PREFIX}-basic-{str(uuid.uuid4())[:8]}"
    create_test_stack(cfn_client, stack_name)
    
    try:
        # Delete using SDK (should auto-confirm)
        HpClusterStack.delete(
            stack_name=stack_name,
            region=REGION
        )
        
        # Wait for deletion to complete
        wait_for_stack_delete_complete(cfn_client, stack_name)
        
        # Verify stack is deleted
        with pytest.raises(Exception) as exc_info:
            cfn_client.describe_stacks(StackName=stack_name)
        assert "does not exist" in str(exc_info.value)
        
    except Exception:
        # Cleanup in case of test failure
        try:
            cfn_client.delete_stack(StackName=stack_name)
        except:
            pass
        raise


# --- test/integration_tests/inference/cli/test_cli_custom_fsx_inference.py ---

def test_custom_create(runner, custom_endpoint_name):
    result = runner.invoke(custom_create, [
        "--namespace", NAMESPACE,
        "--version", VERSION,
        "--instance-type", "ml.c5.2xlarge",
        "--model-name", "test-model-integration-cli-fsx",
        "--model-source-type", "fsx",
        "--model-location", "hf-eqa",
        "--fsx-file-system-id", FSX_LOCATION,
        "--image-uri", "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:2.3.0-transformers4.48.0-cpu-py311-ubuntu22.04",
        "--container-port", "8080",
        "--model-volume-mount-name", "model-weights",
        "--endpoint-name", custom_endpoint_name,
        "--resources-requests", '{"cpu": "3200m", "nvidia.com/gpu": 0, "memory": "12Gi"}',
        "--resources-limits", '{"nvidia.com/gpu": 0}',
        "--env", '{ "SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code", "SAGEMAKER_CONTAINER_LOG_LEVEL": "20", "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600", "ENDPOINT_SERVER_TIMEOUT": "3600", "MODEL_CACHE_ROOT": "/opt/ml/model", "SAGEMAKER_ENV": "1", "SAGEMAKER_MODEL_SERVER_WORKERS": "1" }'
    ])
    assert result.exit_code == 0, result.output

def test_custom_list(runner, custom_endpoint_name):
    result = runner.invoke(custom_list, ["--namespace", NAMESPACE])
    assert result.exit_code == 0
    assert custom_endpoint_name in result.output

def test_custom_describe(runner, custom_endpoint_name):
    result = runner.invoke(custom_describe, [
        "--name", custom_endpoint_name,
        "--namespace", NAMESPACE,
        "--full"
    ])
    assert result.exit_code == 0
    assert custom_endpoint_name in result.output

def test_wait_until_inservice(custom_endpoint_name):
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{custom_endpoint_name}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPEndpoint.get(name=custom_endpoint_name, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return
            
            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")

def test_custom_invoke(runner, custom_endpoint_name):
    result = runner.invoke(custom_invoke, [
        "--endpoint-name", custom_endpoint_name,
        "--body", '{"question" :"what is the name of the planet?", "context":"mars"}',
        "--content-type", "application/list-text"
    ])
    assert result.exit_code == 0
    assert "error" not in result.output.lower()

def test_custom_get_operator_logs(runner):
    result = runner.invoke(custom_get_operator_logs, ["--since-hours", "1"])
    assert result.exit_code == 0

def test_custom_list_pods(runner):
    result = runner.invoke(custom_list_pods, ["--namespace", NAMESPACE])
    assert result.exit_code == 0

def test_custom_delete(runner, custom_endpoint_name):
    result = runner.invoke(custom_delete, [
        "--name", custom_endpoint_name,
        "--namespace", NAMESPACE
    ])
    assert result.exit_code == 0


# --- test/integration_tests/inference/cli/test_cli_custom_s3_inference.py ---

def test_custom_create(runner, custom_endpoint_name):
    result = runner.invoke(custom_create, [
        "--namespace", NAMESPACE,
        "--version", VERSION,
        "--instance-type", "ml.c5.2xlarge",
        "--model-name", "test-model-integration-cli-s3",
        "--model-source-type", "s3",
        "--model-location", "hf-eqa",
        "--s3-bucket-name", BUCKET_LOCATION,
        "--s3-region", REGION,
        "--image-uri", "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:2.3.0-transformers4.48.0-cpu-py311-ubuntu22.04",
        "--container-port", "8080",
        "--model-volume-mount-name", "model-weights",
        "--endpoint-name", custom_endpoint_name,
        "--resources-requests", '{"cpu": "3200m", "nvidia.com/gpu": 0, "memory": "12Gi"}',
        "--resources-limits", '{"nvidia.com/gpu": 0}',
        "--env", '{ "SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code", "SAGEMAKER_CONTAINER_LOG_LEVEL": "20", "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600", "ENDPOINT_SERVER_TIMEOUT": "3600", "MODEL_CACHE_ROOT": "/opt/ml/model", "SAGEMAKER_ENV": "1", "SAGEMAKER_MODEL_SERVER_WORKERS": "1" }'
    ])
    assert result.exit_code == 0, result.output

def test_custom_list(runner, custom_endpoint_name):
    result = runner.invoke(custom_list, ["--namespace", NAMESPACE])
    assert result.exit_code == 0
    assert custom_endpoint_name in result.output

def test_custom_describe(runner, custom_endpoint_name):
    result = runner.invoke(custom_describe, [
        "--name", custom_endpoint_name,
        "--namespace", NAMESPACE,
        "--full"
    ])
    assert result.exit_code == 0
    assert custom_endpoint_name in result.output

def test_wait_until_inservice(custom_endpoint_name):
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{custom_endpoint_name}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPEndpoint.get(name=custom_endpoint_name, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return
            
            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")

def test_custom_invoke(runner, custom_endpoint_name):
    result = runner.invoke(custom_invoke, [
        "--endpoint-name", custom_endpoint_name,
        "--body", '{"question" :"what is the name of the planet?", "context":"mars"}',
        "--content-type", "application/list-text"
    ])
    assert result.exit_code == 0
    assert "error" not in result.output.lower()

def test_custom_get_operator_logs(runner):
    result = runner.invoke(custom_get_operator_logs, ["--since-hours", "1"])
    assert result.exit_code == 0

def test_custom_list_pods(runner):
    result = runner.invoke(custom_list_pods, ["--namespace", NAMESPACE])
    assert result.exit_code == 0

def test_custom_delete(runner, custom_endpoint_name):
    result = runner.invoke(custom_delete, [
        "--name", custom_endpoint_name,
        "--namespace", NAMESPACE
    ])
    assert result.exit_code == 0


# --- test/integration_tests/inference/cli/test_cli_jumpstart_inference.py ---

def test_js_create(runner, js_endpoint_name):
    result = runner.invoke(js_create, [
        "--namespace", NAMESPACE,
        "--version", VERSION,
        "--model-id", "deepseek-llm-r1-distill-qwen-1-5b",
        "--instance-type", "ml.g5.8xlarge",
        "--endpoint-name", js_endpoint_name,
    ])
    assert result.exit_code == 0, result.output

def test_js_list(runner, js_endpoint_name):
    result = runner.invoke(js_list, ["--namespace", NAMESPACE])
    assert result.exit_code == 0
    assert js_endpoint_name in result.output

def test_js_describe(runner, js_endpoint_name):
    result = runner.invoke(js_describe, [
        "--name", js_endpoint_name,
        "--namespace", NAMESPACE,
        "--full"
    ])
    assert result.exit_code == 0
    assert js_endpoint_name in result.output

def test_wait_until_inservice(js_endpoint_name):
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{js_endpoint_name}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPJumpStartEndpoint.get(name=js_endpoint_name, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return

            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")

def test_custom_invoke(runner, js_endpoint_name):
    result = runner.invoke(custom_invoke, [
        "--endpoint-name", js_endpoint_name,
        "--body", '{"inputs": "What is the capital of USA?"}'
    ])
    assert result.exit_code == 0
    assert "error" not in result.output.lower()

def test_js_get_operator_logs(runner):
    result = runner.invoke(js_get_operator_logs, ["--since-hours", "1"])
    assert result.exit_code == 0

def test_js_list_pods(runner):
    result = runner.invoke(js_list_pods, ["--namespace", NAMESPACE])
    assert result.exit_code == 0

def test_js_delete(runner, js_endpoint_name):
    result = runner.invoke(js_delete, [
        "--name", js_endpoint_name,
        "--namespace", NAMESPACE
    ])
    assert result.exit_code == 0


# --- test/integration_tests/inference/cli/test_cli_jumpstart_inference_with_mig.py ---

def test_js_create(runner, js_endpoint_name):
    result = runner.invoke(js_create, [
        "--namespace", NAMESPACE,
        "--version", VERSION,
        "--model-id", "deepseek-llm-r1-distill-qwen-1-5b",
        "--instance-type", "ml.p4d.24xlarge",
        "--endpoint-name", js_endpoint_name,
        "--accelerator-partition-type", "mig-7g.40gb",
        "--accelerator-partition-validation", "true",
    ])
    assert result.exit_code == 0, result.output

def test_js_list(runner, js_endpoint_name):
    result = runner.invoke(js_list, ["--namespace", NAMESPACE])
    assert result.exit_code == 0
    assert js_endpoint_name in result.output

def test_js_describe(runner, js_endpoint_name):
    result = runner.invoke(js_describe, [
        "--name", js_endpoint_name,
        "--namespace", NAMESPACE,
        "--full"
    ])
    assert result.exit_code == 0
    assert js_endpoint_name in result.output

def test_wait_until_inservice(js_endpoint_name):
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{js_endpoint_name}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPJumpStartEndpoint.get(name=js_endpoint_name, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return

            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")

def test_custom_invoke(runner, js_endpoint_name):
    result = runner.invoke(custom_invoke, [
        "--endpoint-name", js_endpoint_name,
        "--body", '{"inputs": "What is the capital of USA?"}'
    ])
    assert result.exit_code == 0
    assert "error" not in result.output.lower()

def test_js_get_operator_logs(runner):
    result = runner.invoke(js_get_operator_logs, ["--since-hours", "1"])
    assert result.exit_code == 0

def test_js_list_pods(runner):
    result = runner.invoke(js_list_pods, ["--namespace", NAMESPACE])
    assert result.exit_code == 0

def test_js_delete(runner, js_endpoint_name):
    result = runner.invoke(js_delete, [
        "--name", js_endpoint_name,
        "--namespace", NAMESPACE
    ])
    assert result.exit_code == 0


# --- test/integration_tests/inference/sdk/test_sdk_custom_fsx_inference.py ---

def test_wait_until_inservice():
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{ENDPOINT_NAME}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPEndpoint.get(name=ENDPOINT_NAME, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return
            
            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")


# --- test/integration_tests/inference/sdk/test_sdk_custom_s3_inference.py ---

def test_wait_until_inservice():
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{ENDPOINT_NAME}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPEndpoint.get(name=ENDPOINT_NAME, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return
            
            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")


# --- test/integration_tests/inference/sdk/test_sdk_jumpstart_inference.py ---

def test_wait_until_inservice():
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{ENDPOINT_NAME}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPJumpStartEndpoint.get(name=ENDPOINT_NAME, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return

            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")


# --- test/integration_tests/inference/sdk/test_sdk_jumpstart_inference_with_mig.py ---

def test_wait_until_inservice():
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{ENDPOINT_NAME}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPJumpStartEndpoint.get(name=ENDPOINT_NAME, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return

            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")


# --- test/integration_tests/init/test_custom_creation.py ---

def test_init_custom(runner, custom_endpoint_name, test_directory):
    """Initialize custom endpoint template and verify file creation."""
    result = runner.invoke(
        init, ["hyp-custom-endpoint", "."], catch_exceptions=False
    )
    assert_command_succeeded(result)
    assert_init_files_created("./", "hyp-custom-endpoint")

def test_validate_custom(runner, custom_endpoint_name, test_directory):
    """Validate custom endpoint configuration for correctness."""
    result = runner.invoke(validate, [], catch_exceptions=False)
    assert_command_succeeded(result)

def test_create_custom(runner, custom_endpoint_name, test_directory):
    """Create custom endpoint for deployment and verify template rendering."""
    result = runner.invoke(create, [], catch_exceptions=False)
    assert_command_succeeded(result)

    # Verify expected submission messages appear  
    assert "Submitted!" in result.output
    assert "Creating sagemaker model and endpoint" in result.output
    assert custom_endpoint_name in result.output
    assert "The process may take a few minutes" in result.output

def test_wait_until_inservice(custom_endpoint_name, test_directory):
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{custom_endpoint_name}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPEndpoint.get(name=custom_endpoint_name, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return
            
            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")

def test_custom_invoke(runner, custom_endpoint_name, test_directory):
    result = runner.invoke(custom_invoke, [
        "--endpoint-name", custom_endpoint_name,
        "--body", '{"question" :"what is the name of the planet?", "context":"mars"}',
        "--content-type", "application/list-text"
    ])
    assert result.exit_code == 0
    assert "error" not in result.output.lower()

def test_custom_delete(runner, custom_endpoint_name, test_directory):
    """Clean up deployed custom endpoint using CLI delete command."""
    result = runner.invoke(delete, [
        "hyp-custom-endpoint",
        "--name", custom_endpoint_name,
        "--namespace", NAMESPACE
    ])
    assert_command_succeeded(result)


# --- test/integration_tests/init/test_jumpstart_creation.py ---

def test_init_jumpstart(runner, js_endpoint_name, test_directory):
    """Initialize JumpStart endpoint template and verify file creation."""
    result = runner.invoke(
        init, ["hyp-jumpstart-endpoint", "."], catch_exceptions=False
    )
    assert_command_succeeded(result)
    assert_init_files_created("./", "hyp-jumpstart-endpoint")

def test_validate_jumpstart(runner, js_endpoint_name, test_directory):
    """Validate JumpStart endpoint configuration for correctness."""
    result = runner.invoke(validate, [], catch_exceptions=False)
    assert_command_succeeded(result)

def test_create_jumpstart(runner, js_endpoint_name, test_directory):
    """Create JumpStart endpoint for deployment and verify template rendering."""
    result = runner.invoke(create, [], catch_exceptions=False)
    assert_command_succeeded(result)

    assert "Submitted!" in result.output

def test_wait_until_inservice(js_endpoint_name, test_directory):
    """Poll SDK until specific JumpStart endpoint reaches DeploymentComplete"""
    print(f"[INFO] Waiting for JumpStart endpoint '{js_endpoint_name}' to be DeploymentComplete...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking endpoint status...")

        try:
            ep = HPJumpStartEndpoint.get(name=js_endpoint_name, namespace=NAMESPACE)
            state = ep.status.endpoints.sagemaker.state
            print(f"[DEBUG] Current state: {state}")
            if state == "CreationCompleted":
                print("[INFO] Endpoint is in CreationCompleted state.")
                return

            deployment_state = ep.status.deploymentStatus.deploymentObjectOverallState
            if deployment_state == "DeploymentFailed":
                pytest.fail("Endpoint deployment failed.")

        except Exception as e:
            print(f"[ERROR] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail("[ERROR] Timed out waiting for endpoint to be DeploymentComplete")

def test_custom_invoke(runner, js_endpoint_name, test_directory):
    result = runner.invoke(custom_invoke, [
        "--endpoint-name", js_endpoint_name,
        "--body", '{"inputs": "What is the capital of USA?"}'
    ])
    assert result.exit_code == 0
    assert "error" not in result.output.lower()

def test_js_delete(runner, js_endpoint_name, test_directory):
    """Clean up deployed JumpStart endpoint using CLI delete command."""
    result = runner.invoke(delete, [
        "hyp-jumpstart-endpoint",
        "--name", js_endpoint_name,
        "--namespace", NAMESPACE
    ])
    assert_command_succeeded(result)


# --- test/integration_tests/init/test_pytorch_job_creation.py ---

def test_init_pytorch_job(runner, pytorch_job_name, test_directory):
    """Initialize PyTorch job template and verify file creation."""
    result = runner.invoke(
        init, ["hyp-pytorch-job", "."], catch_exceptions=False
    )
    assert_command_succeeded(result)
    assert_init_files_created("./", "hyp-pytorch-job")

def test_validate_pytorch_job(runner, pytorch_job_name, test_directory):
    """Validate PyTorch job configuration for correctness."""
    result = runner.invoke(validate, [], catch_exceptions=False)
    assert_command_succeeded(result)

def test_create_pytorch_job(runner, pytorch_job_name, test_directory):
    """Create PyTorch job for deployment and verify template rendering."""
    result = runner.invoke(create, [], catch_exceptions=False)
    assert_command_succeeded(result)
                             
    # Verify expected submission messages appear
    assert "Submitted!" in result.output
    assert "Successfully submitted HyperPodPytorchJob" in result.output
    assert pytorch_job_name in result.output

def test_wait_for_job_running(pytorch_job_name, test_directory):
    """Poll SDK until PyTorch job reaches Running state."""
    print(f"[INFO] Waiting for PyTorch job '{pytorch_job_name}' to be Running...")
    deadline = time.time() + (TIMEOUT_MINUTES * 60)
    poll_count = 0

    while time.time() < deadline:
        poll_count += 1
        print(f"[DEBUG] Poll #{poll_count}: Checking job status...")

        try:
            job = HyperPodPytorchJob.get(name=pytorch_job_name, namespace=NAMESPACE)
            if job.status and hasattr(job.status, 'conditions'):
                # Check for Running condition
                for condition in job.status.conditions:
                    if condition.type in ["PodsRunning", "Running"] and condition.status == "True":
                        print(f"[INFO] Job {pytorch_job_name} is now Running")
                        return
                    elif condition.type == "Failed" and condition.status == "True":
                        pytest.fail(f"Job {pytorch_job_name} failed: {condition.reason}")
                
                print(f"[DEBUG] Job status conditions: {[c.type for c in job.status.conditions]}")
            else:
                print(f"[DEBUG] Job status not yet available")

        except Exception as e:
            print(f"[DEBUG] Exception during polling: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail(f"[ERROR] Timed out waiting for job {pytorch_job_name} to be Running")

def test_pytorch_job_delete(pytorch_job_name, test_directory):
    """Clean up deployed PyTorch job using CLI delete command and verify deletion."""
    delete_result = execute_command([
        "hyp", "delete", "hyp-pytorch-job",
        "--job-name", pytorch_job_name,
        "--namespace", NAMESPACE
    ])
    assert delete_result.returncode == 0
    print(f"[INFO] Successfully deleted job: {pytorch_job_name}")

    # Wait a moment for the job to be deleted
    time.sleep(5)

    # Verify the job is no longer listed
    list_result = execute_command(["hyp", "list", "hyp-pytorch-job"])
    assert list_result.returncode == 0

    # The job name should no longer be in the output
    assert pytorch_job_name not in list_result.stdout
    print(f"[INFO] Verified job {pytorch_job_name} is no longer listed after deletion")

