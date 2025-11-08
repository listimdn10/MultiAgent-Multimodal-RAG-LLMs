# crew_run.py
from crewai import Crew, Process
from agent_fusion import build_fusion_agent
from explainer import build_explainer_agent



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