# AI/ML Use Case Generator and Resource Collector

## Overview
An advanced multi-agent system that generates AI/ML use cases and collects implementation resources using CrewAI and LangChain. The system analyzes companies/industries, generates relevant use cases, and provides detailed implementation resources and recommendations.

## 🌟 Key Features
- Industry and Company Analysis
- AI/ML Use Case Generation
- Resource Collection and Categorization
- Technical Implementation Planning
- Comprehensive Documentation Generation

## 🏗️ Architecture

## 🛠️ Technical Stack
- **Framework**: CrewAI for multi-agent orchestration
- **LLM Integration**: Gemini-2.0-flash-exp via Gemini api
- **Search Capability**: SerperDev API for web search
- **Development**: Python 3.8+
-  **API Keys for:
  - Google Generative AI (Gemini)
  - Serper.dev for search capabilities
- **Key Libraries**: LangChain,CrewAI, Pydantic, Tenacity

## 📋 Agents Overview

### 1. Industry Research Agent
**Purpose**: Analyzes company/industry information and generates comprehensive insights
**Capabilities**:
- Company profile analysis
- Industry sector research
- Technology stack assessment
- Market position evaluation
- Competitive analysis

### 2. Market Analysis Agent
**Purpose**: Generates AI/ML use cases based on industry analysis
**Capabilities**:
- Use case identification
- Technical requirement analysis
- Implementation planning
- ROI assessment
- Priority matrix generation

### 3. Resource Collection Agent
**Purpose**: Collects and categorizes implementation resources
**Capabilities**:
- GitHub repository discovery
- Dataset identification
- Tutorial collection
- Tool/framework recommendations
- Implementation guide generation

### 4. **Orchestrator**
   - Manages task execution and data flow between agents.
   - Saves results in JSON and CSV formats for documentation and reporting.


 ### 5. Results 
   - Example industry research output.
   - Proposed AI/ML use cases.
   - Example resources collected.
   - Output formats (JSON and CSV).
 


 
