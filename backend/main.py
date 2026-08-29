"""
AI 旅游规划助手 — FastAPI 后端
==============================
调用 DeepSeek Chat API，把用户 5 步输入转成一份结构化的可执行行程包。

启动:
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 本地开发请用 .env 或环境变量
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

app = FastAPI(
    title="AI Travel Planner",
    version="1.0.0",
    description="5 步输入 → AI 生成完整行程包",
)

# 开放 CORS 给本地前端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 请求 / 响应 Schema
# ---------------------------------------------------------------------------
class TravelInput(BaseModel):
    departure: str = Field(..., description="出发地，例如 杭州")
    destination: str = Field(..., description="目的地，例如 成都")
    dates: str = Field(..., description="出行日期，例如 国庆 10.1-10.4")
    days: int = Field(..., ge=1, le=14, description="游玩天数")
    budget: str = Field(..., description="预算区间，例如 3000-5000元")
    companions: str = Field("一个人", description="同行人，例如 2人情侣 / 带老人孩子")
    pace: str = Field("轻松", description="节奏：轻松 / 正常 / 紧凑")
    interests: List[str] = Field(default_factory=list, description="偏好标签")
    avoid: List[str] = Field(default_factory=list, description="忌口 / 避开项")


class ItineraryItem(BaseModel):
    time: str = Field(..., description="时间段，例如 09:00-11:00")
    title: str
    type: str = Field(..., description="景点/美食/交通/住宿")
    location: Optional[str] = None
    duration: Optional[str] = None
    cost: Optional[str] = None
    tips: Optional[str] = None


class DayPlan(BaseModel):
    day: int
    theme: str = Field(..., description="主题，例如 经典老城 + 美食")
    items: List[ItineraryItem]


class ChecklistItem(BaseModel):
    category: str = Field(..., description="天气预报/穿搭/交通/门票/避坑")
    title: str
    content: str


class TripPlan(BaseModel):
    trip_summary: str = Field(..., description="一句话概述")
    estimated_total_cost: str = Field(..., description="预估总花费")
    best_transport: str = Field(..., description="推荐交通方式")
    accommodation_tip: str = Field(..., description="住宿建议")
    days: List[DayPlan]
    checklist: List[ChecklistItem]
    generated_at: str


# ---------------------------------------------------------------------------
# Prompt 模板 —— 核心灵魂
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
你是一位资深的旅行规划专家，拥有 10 年经验，熟悉中国及东南亚热门目的地的美食、景点、交通和避坑指南。

你的任务：根据用户给出的 5 步输入，生成一份 **可直接执行** 的行程包。

硬性要求：
1. 路线必须 **合理** —— 同一天的景点尽量在同一区域，减少折返
2. 时间必须 **真实** —— 景点开放时间、游玩时长、交通时间都要符合实际
3. 预算必须 **诚实** —— 各项花费要有依据，总花费要在用户预算区间内
4. 注意事项要 **具体** —— 不能写"注意天气"，要写"国庆成都多雨，带折叠伞 + 穿防水鞋"
5. 风格要 **接地气** —— 用口语化中文，像一个靠谱朋友给的建议
6. 只输出 JSON，不要任何解释、markdown 代码块、或多余文字
"""

USER_PROMPT_TEMPLATE = """\
【用户 5 步输入】
1. 出发地：{departure}
2. 目的地：{destination}
3. 时间：{dates}（{days} 天）
4. 预算：{budget}
5. 同行人：{companions}
6. 节奏偏好：{pace}
7. 兴趣标签：{interests}
8. 忌口 / 避开项：{avoid}

【请输出 JSON 结构，严格符合以下 Schema】
{{
  "trip_summary": "一句话概述这次旅行的亮点（30字以内）",
  "estimated_total_cost": "总花费，例如 约 4200 元（含机酒）",
  "best_transport": "推荐交通方式，例如 高铁直达 4 小时",
  "accommodation_tip": "住宿建议，例如 推荐住春熙路地铁站步行 5 分钟范围内",
  "days": [
    {{
      "day": 1,
      "theme": "当天主题，例如 经典老城 + 美食",
      "items": [
        {{
          "time": "09:00-11:00",
          "title": "宽窄巷子",
          "type": "景点",
          "location": "青羊区长顺上街",
          "duration": "2 小时",
          "cost": "免费",
          "tips": "不要在门口买锦里小吃，往里走 200 米有本地人的店"
        }}
      ]
    }}
  ],
  "checklist": [
    {{
      "category": "天气预报",
      "title": "国庆成都天气",
      "content": "10 月上旬多雨，18-25℃，带折叠伞 + 薄外套"
    }},
    {{
      "category": "穿搭建议",
      "title": "鞋子优先",
      "content": "成都景点之间靠地铁 + 步行，穿轻便运动鞋"
    }},
    {{
      "category": "交通出行",
      "title": "市内交通",
      "content": "推荐买成都地铁日卡 10 元/天，覆盖主要景点"
    }},
    {{
      "category": "门票预订",
      "title": "热门景点",
      "content": "熊猫基地必须提前 1 天在官微预约，国博需身份证免费进"
    }},
    {{
      "category": "避坑指南",
      "title": "美食避坑",
      "content": "春熙路网红店排队 1 小时起步，附近小区楼下的苍蝇馆子更地道"
    }}
  ]
}}

现在开始生成。记住：**只输出 JSON，不要任何其他文字**。"""


# ---------------------------------------------------------------------------
# 调用 DeepSeek
# ---------------------------------------------------------------------------
async def call_deepseek(input_data: TravelInput) -> Dict[str, Any]:
    prompt = USER_PROMPT_TEMPLATE.format(
        departure=input_data.departure,
        destination=input_data.destination,
        dates=input_data.dates,
        days=input_data.days,
        budget=input_data.budget,
        companions=input_data.companions,
        pace=input_data.pace,
        interests="、".join(input_data.interests) if input_data.interests else "无特别偏好",
        avoid="、".join(input_data.avoid) if input_data.avoid else "无",
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 4000,
                "response_format": {"type": "json_object"},
            },
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek API 错误 {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    return _parse_json(raw)


def _parse_json(raw: str) -> Dict[str, Any]:
    """DeepSeek 偶尔会在 JSON 外面套 ```json ... ```，这里剥干净。"""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # 尝试从第一个 { 开始截取
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise HTTPException(status_code=502, detail=f"LLM 返回了无法解析的 JSON: {e}")


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "ts": datetime.now().isoformat()}


@app.post("/api/generate", response_model=TripPlan)
async def generate_plan(data: TravelInput):
    """根据 5 步输入生成完整行程包"""
    if not data.destination.strip():
        raise HTTPException(status_code=400, detail="目的地不能为空")

    raw = await call_deepseek(data)
    raw["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 让 Pydantic 做一次校验（失败就返回原始 JSON 但不 crash）
    try:
        return TripPlan(**raw)
    except Exception as e:
        # 宽松返回，前端可以容错
        raw.setdefault("trip_summary", "")
        raw.setdefault("estimated_total_cost", "")
        raw.setdefault("best_transport", "")
        raw.setdefault("accommodation_tip", "")
        raw.setdefault("days", [])
        raw.setdefault("checklist", [])
        raw.setdefault("generated_at", "")
        return raw


@app.get("/api/demo")
async def demo_plan():
    """返回一份示例行程，方便前端调试，不用每次调 LLM"""
    sample = {
        "trip_summary": "3 天 2 夜 · 成都慢生活 · 熊猫 + 火锅 + 宽窄巷子",
        "estimated_total_cost": "约 4200 元（2 人，含往返高铁 + 酒店）",
        "best_transport": "杭州东 → 成都东 高铁直达 7.5 小时，二等座 ¥780/人",
        "accommodation_tip": "推荐住春熙路地铁 2/3 号线附近，步行 5 分钟到 IFS",
        "days": [
            {
                "day": 1,
                "theme": "抵达 + 老城初印象",
                "items": [
                    {
                        "time": "08:00-15:30",
                        "title": "高铁杭州东 → 成都东",
                        "type": "交通",
                        "location": "杭州东站",
                        "duration": "7.5 小时",
                        "cost": "¥780/人",
                        "tips": "建议选靠窗 A 座，看长江沿线风景；车上自备零食"
                    },
                    {
                        "time": "16:30-17:30",
                        "title": "酒店入住 + 休息",
                        "type": "住宿",
                        "location": "春熙路附近",
                        "duration": "30 分钟",
                        "cost": "¥350/晚",
                        "tips": "推荐亚朵 / 桔子水晶，步行 5 分钟到地铁 2 号线"
                    },
                    {
                        "time": "18:30-21:00",
                        "title": "建设路小吃街",
                        "type": "美食",
                        "location": "成华区建设路",
                        "duration": "2.5 小时",
                        "cost": "¥80/人",
                        "tips": "必吃：烤脑花、钵钵鸡、冰粉；避开网红店排队"
                    }
                ]
            },
            {
                "day": 2,
                "theme": "熊猫基地 + 宽窄巷子",
                "items": [
                    {
                        "time": "07:30-11:00",
                        "title": "大熊猫繁育研究基地",
                        "type": "景点",
                        "location": "成华区熊猫大道 1375 号",
                        "duration": "3.5 小时",
                        "cost": "¥55 门票",
                        "tips": "必须提前 1 天在官微预约！上午 8 点到，熊猫最活跃"
                    },
                    {
                        "time": "12:00-13:30",
                        "title": "陈麻婆豆腐（总店）",
                        "type": "美食",
                        "location": "青羊区青华路 12 号",
                        "duration": "1.5 小时",
                        "cost": "¥80/人",
                        "tips": "总店最正宗，麻婆豆腐 + 夫妻肺片组合"
                    },
                    {
                        "time": "14:30-17:00",
                        "title": "宽窄巷子",
                        "type": "景点",
                        "location": "青羊区长顺上街",
                        "duration": "2.5 小时",
                        "cost": "免费",
                        "tips": "往里走 200 米有本地人的茶馆，盖碗茶 ¥15"
                    },
                    {
                        "time": "18:30-20:30",
                        "title": "春熙路 + IFS 打卡",
                        "type": "景点",
                        "location": "锦江区春熙路",
                        "duration": "2 小时",
                        "cost": "免费",
                        "tips": "IFS 顶楼大熊猫屁股是必拍机位，从商场内部上 7 楼"
                    }
                ]
            },
            {
                "day": 3,
                "theme": "锦里 + 返程",
                "items": [
                    {
                        "time": "09:00-11:30",
                        "title": "锦里古街",
                        "type": "景点",
                        "location": "武侯区锦里中路",
                        "duration": "2.5 小时",
                        "cost": "免费",
                        "tips": "早上人少好拍照；可以给朋友带点蜀绣小挂件"
                    },
                    {
                        "time": "12:00-13:30",
                        "title": "蜀大侠火锅",
                        "type": "美食",
                        "location": "锦江区红星路三段",
                        "duration": "1.5 小时",
                        "cost": "¥150/人",
                        "tips": "微辣就够了！配解辣神器豆奶"
                    },
                    {
                        "time": "14:30-22:00",
                        "title": "高铁成都东 → 杭州东",
                        "type": "交通",
                        "location": "成都东站",
                        "duration": "7.5 小时",
                        "cost": "¥780/人",
                        "tips": "车上早点订票；带个 U 型枕；可以看看书或者打个盹"
                    }
                ]
            }
        ],
        "checklist": [
            {
                "category": "天气预报",
                "title": "国庆成都天气",
                "content": "10 月上旬多雨，18-25℃，带折叠伞 + 薄外套"
            },
            {
                "category": "穿搭建议",
                "title": "鞋子优先",
                "content": "成都景点之间靠地铁 + 步行，穿轻便运动鞋"
            },
            {
                "category": "交通出行",
                "title": "市内交通",
                "content": "推荐买成都地铁日卡 10 元/天，覆盖主要景点"
            },
            {
                "category": "门票预订",
                "title": "热门景点",
                "content": "熊猫基地必须提前 1 天在官微预约，国博需身份证免费进"
            },
            {
                "category": "避坑指南",
                "title": "美食避坑",
                "content": "春熙路网红店排队 1 小时起步，附近小区楼下的苍蝇馆子更地道"
            }
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return TripPlan(**sample)


# ---------------------------------------------------------------------------
# 入口 —— 本地开发 / Render / Hugging Face Spaces 通用
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
