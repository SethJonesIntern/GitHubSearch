# YUHAO-corn/manufacturing-agents
# 3 LLM-backed test functions across 103 test files
# Source: https://github.com/YUHAO-corn/manufacturing-agents

# --- tests/test_manufacturing_end_to_end.py ---

def test_preprocessing_assistant():
    """测试1：预处理助手参数生成"""
    print("🔍 测试1：预处理助手参数生成")
    print("=" * 60)
    
    try:
        from manufacturingagents.manufacturingagents.utils.preprocessing_assistant import PreprocessingAssistant
        
        assistant = PreprocessingAssistant(model_provider="dashscope")
        
        # 生成API参数
        api_params = assistant.generate_api_parameters(
            city_name="广州市",
            brand_name="美的", 
            product_type="空调",
            special_focus="关注天气影响和季节性需求"
        )
        
        print(f"✅ 预处理助手成功生成参数")
        print(f"   参数数量: {len(api_params)} 个API")
        print(f"   包含API: {list(api_params.keys())}")
        
        return api_params
        
    except Exception as e:
        print(f"❌ 预处理助手测试失败: {str(e)}")
        return None

def test_individual_tools():
    """测试2：单独测试每个工具函数"""
    print("\n🛠️ 测试2：单独测试工具函数")
    print("=" * 60)
    
    try:
        from manufacturingagents.agents.utils.agent_utils import Toolkit
        
        toolkit = Toolkit()
        
        # 测试天气工具
        print("🌤️ 测试天气工具...")
        weather_result = toolkit.get_manufacturing_weather_data.invoke({"city_name": "广州"})
        print(f"   天气数据长度: {len(weather_result)} 字符")
        print(f"   状态: {'✅ 成功' if '❌' not in weather_result else '❌ 失败'}")
        
        # 测试PMI工具  
        print("📈 测试PMI工具...")
        pmi_result = toolkit.get_manufacturing_pmi_data.invoke({"time_range": "最近3个月"})
        print(f"   PMI数据长度: {len(pmi_result)} 字符")
        print(f"   状态: {'✅ 成功' if '❌' not in pmi_result else '❌ 失败'}")
        
        # 测试新闻工具
        print("📰 测试新闻工具...")
        news_result = toolkit.get_manufacturing_news_data.invoke({"query_params": "广州美的空调"})
        print(f"   新闻数据长度: {len(news_result)} 字符")
        print(f"   状态: {'✅ 成功' if '❌' not in news_result else '❌ 失败'}")
        
        return {
            'weather': weather_result,
            'pmi': pmi_result, 
            'news': news_result
        }
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {str(e)}")
        return None


# --- tests/manufacturing/test_manufacturing_react.py ---

def test_react_components():
    """测试ReAct相关组件"""
    print("\n🔧 === ReAct组件测试 ===")
    
    # 测试Tongyi LLM
    print("\n🧠 测试Tongyi LLM...")
    try:
        from langchain_community.llms import Tongyi
        llm = Tongyi()
        llm.model_name = "qwen-turbo"
        print("✅ Tongyi LLM初始化成功")
    except Exception as e:
        print(f"❌ Tongyi LLM初始化失败: {e}")
        return False
    
    # 测试ReAct Agent创建
    print("\n🤖 测试ReAct Agent创建...")
    try:
        from langchain.agents import create_react_agent
        from langchain import hub
        from langchain_core.tools import BaseTool
        
        # 测试工具
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "测试工具"
            
            def _run(self, query: str = "") -> str:
                return "测试工具调用成功"
        
        tools = [TestTool()]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        print("✅ ReAct Agent创建成功")
        
    except Exception as e:
        print(f"❌ ReAct Agent创建失败: {e}")
        return False
    
    # 测试工具包
    print("\n🛠️ 测试制造业工具包...")
    try:
        from manufacturingagents.agents.utils.agent_utils import Toolkit
        toolkit = Toolkit()
        
        # 测试制造业工具
        result = toolkit.get_manufacturing_pmi_data.invoke({"time_range": "最近3个月"})
        print(f"✅ 制造业工具包测试成功: {len(result)} 字符")
    except Exception as e:
        print(f"❌ 制造业工具包测试失败: {e}")
        return False
    
    print("✅ 所有ReAct组件测试通过")
    return True

