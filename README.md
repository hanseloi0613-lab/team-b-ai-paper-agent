# TEAM B - AI Paper Draft Agent

## 프로젝트 목표

사용자가 우주 AI / 우주시스템 자율화 관련 제목 또는 주제를 입력하면,

1. RAG가 관련 논문 및 최신 근거를 검색하고
2. 직접 학습한 Transformer가 학술적 문장을 생성하며
3. 제한된 Finalizer가 근거를 바탕으로
   서론 / 본론 / 결론 구조의 약 4,500자 논문 초안을 생성하는 프로젝트입니다.

---

## 연구 주제 3축

### 1. rover_autonomy
행성 로버 자율항법
- Path Planning
- Obstacle / Hazard Avoidance
- Localization
- Scientific Target Selection

### 2. onboard_ai
심우주 Onboard AI
- Communication Delay
- Autonomous Decision Making
- Navigation
- Mission Planning

### 3. satellite_autonomy
위성 / 우주시스템 자율운영
- Fault Detection
- Fault Diagnosis
- Recovery
- Autonomous Operations

---

# 데이터 파이프라인

```text
Collector
    ↓
Resolver
    ↓
Parser
    ↓
Cleaner
    ↓
Quality / Hash / Dedup
    ↓
AWS RDS PostgreSQL