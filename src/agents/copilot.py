import pandas as pd

class DecisionAgent:
    """Analyzes the numerical gaps and decides what went wrong."""
    def analyze(self, employee_data, predicted_band):
        issues = []
        if employee_data['training_hours'] < 10:
            issues.append("Low training hours")
        if employee_data['bug_count'] > 5 and employee_data['department'] == 'Engineering':
            issues.append("High bug generation rate")
        if employee_data['on_time_delivery_rate'] < 0.75:
            issues.append("Consistent task delays")
        if employee_data['avg_login_hours'] < 6:
            issues.append("Low daily engagement/login hours")
        
        return issues

class AutomationAgent:
    """Automates HR actions/recommendations based on identified issues."""
    def trigger_actions(self, issues):
        actions = []
        for index, issue in enumerate(issues, start=1):
            if issue == "Low training hours":
                actions.append(f"{index}. System Action: Auto-enrolling in Level 1 Certification Course.")
            elif issue == "High bug generation rate":
                actions.append(f"{index}. Recommendation: Assign a Pair-Programming Mentor.")
            elif issue == "Consistent task delays":
                actions.append(f"{index}. Recommendation: Enroll manager in Agile Sprint Planning training.")
            elif issue == "Low daily engagement/login hours":
                actions.append(f"{index}. Alert: HR Business Partner to schedule 1-on-1 check-in (Flight Risk).")
        
        if not actions:
            actions.append("1. Continue monitoring. Maintain current trajectory.")
        return actions

class CommunicationAgent:
    """Formats the data into human-readable Copilot outputs for managers."""
    def respond(self, predicted_band, issues, actions):
        if predicted_band == "High":
            tone = "🌟 Excellent Outlook:"
            summary = "This employee is on track for a High performance rating. Keep up the good work and consider promotion readiness."
        elif predicted_band == "Medium":
            tone = "📊 Steady Performace:"
            summary = "This employee is performing at the expected baseline. Look out for opportunities to push them to the next level."
        else:
            tone = "⚠️ Intervention Required:"
            summary = "The AI model has detected a high probability of a 'Low' performance rating in the upcoming cycle."

        response = f"{tone} {summary}\n\n"
        if predicted_band == "Low" or predicted_band == "Medium":
            response += "**Identified Risk Factors:**\n"
            for issue in issues:
                response += f"- {issue}\n"
            response += "\n**Recommended Prescriptive Actions:**\n"
            for act in actions:
                response += f"{act}\n"
        return response

class OptimaHRCopilot:
    """
    Orchestrates the Multi-Agent System.
    In a fully deployed Enterprise scale app, this would use LangChain and LLMs.
    For this deterministic system, we use an Expert Rule-Based emulation.
    """
    def __init__(self):
        self.decision_agent = DecisionAgent()
        self.automation_agent = AutomationAgent()
        self.communication_agent = CommunicationAgent()

    def get_prescriptive_advice(self, employee_data, predicted_band):
        # 1. Decision Agent finds issues
        issues = self.decision_agent.analyze(employee_data, predicted_band)
        
        # 2. Automation Agent determines fixes
        actions = self.automation_agent.trigger_actions(issues)
        
        # 3. Communication Agent generates UI response
        final_advice = self.communication_agent.respond(predicted_band, issues, actions)
        
        return final_advice
