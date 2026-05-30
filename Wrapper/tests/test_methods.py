"""Class method handling: indexing, self.X / cls.X / ClassName.X resolution."""


# Direct match in a method body
def test_basic_method_is_direct(invokers):
    assert "test_repo.methods.basic.AgentClient.chat" in invokers


# self.X resolution
def test_self_chain_helper_is_direct(invokers):
    assert "test_repo.methods.self_chain.AgentClient.helper" in invokers


def test_self_chain_public_is_transitive(invokers):
    qname = "test_repo.methods.self_chain.AgentClient.public_run"
    assert qname in invokers
    assert "calls test_repo.methods.self_chain.AgentClient.helper" in invokers[qname]


# cls.X resolution
def test_classmethod_from_config_is_direct(invokers):
    assert "test_repo.methods.cls_classmethod.AgentClient.from_config" in invokers


def test_classmethod_default_uses_cls(invokers):
    qname = "test_repo.methods.cls_classmethod.AgentClient.make_default"
    assert qname in invokers
    assert "calls test_repo.methods.cls_classmethod.AgentClient.from_config" in invokers[qname]


# ClassName.X resolution
def test_staticmethod_do_work_is_direct(invokers):
    assert "test_repo.methods.classname_call.Helper.do_work" in invokers


def test_classname_call_resolves(invokers):
    qname = "test_repo.methods.classname_call.entry"
    assert qname in invokers
    assert "calls test_repo.methods.classname_call.Helper.do_work" in invokers[qname]
