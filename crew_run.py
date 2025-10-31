# crew_run.py
from unsloth import FastLanguageModel
from crewai import Crew, Process
from agent_fusion import build_fusion_agent
from explainer import build_explainer_agent
import json




def to_json_safe(obj):
    """Chuyển object bất kỳ thành dạng JSON-serializable."""
    try:
        # Nếu object có to_dict()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # Nếu object có raw_output
        elif hasattr(obj, "raw_output"):
            return obj.raw_output
        # Nếu object có __dict__
        elif hasattr(obj, "__dict__"):
            return {k: to_json_safe(v) for k, v in obj.__dict__.items()}
        # Nếu object là list / tuple
        elif isinstance(obj, (list, tuple)):
            return [to_json_safe(i) for i in obj]
        # Nếu object là dict
        elif isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        # Các kiểu còn lại thì convert thành string
        else:
            return str(obj)
    except Exception as e:
        return f"<Unserializable: {e}>"


if __name__ == "__main__":
    fusion_agent, fusion_task = build_fusion_agent()
    explainer_agent, explainer_task = build_explainer_agent()

    crew = Crew(
        agents=[fusion_agent, explainer_agent],
        tasks=[fusion_task, explainer_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    print("\n✅ Kết quả cuối cùng:")
    print(result)

    # crew_run.py (đoạn cuối)
    filename = "multimodal-Audit.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# 🧠 Multi-Modal Audit Result\n\n")
        f.write(repr(result))  # 👈 Lưu toàn bộ object dạng thô

    print(f"\n📁 Kết quả đã được lưu vào file: {filename}")