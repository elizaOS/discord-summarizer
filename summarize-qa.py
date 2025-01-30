from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import dateutil.parser
from typing import Dict, List, Any, Optional
import signal
import sys
import argparse
import json
import os
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from datetime import datetime
from pydantic import BaseModel
from openai import OpenAI

# Initialize console and logging
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    console.print("\n[yellow]Shutting down gracefully...[/]")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class ChatQA(BaseModel):
    question: str
    asker: str
    answer: str
    answerer: str
    context: Optional[str] = None

class ChatAnalysis(BaseModel):
    qa_pairs: List[ChatQA] = []

class DiscordChatAnalyzer:
    def __init__(self, model_name='phi4-chat', model_provider='ollama', 
                 openrouter_model="openai/gpt-3.5-turbo"):
        self._timestamp_cache = {}
        self._user_cache = {}
        self.model_provider = model_provider
        self.openrouter_model = openrouter_model
        
        console.print(Panel.fit(f"[bold cyan]Initializing Discord Q&A Analyzer ({model_provider.upper()})[/]"))
        
        try:
            if model_provider == 'ollama':
                self.model = ChatOllama(
                    model=model_name,
                    temperature=0.2,
                    num_ctx=8192
                )
            elif model_provider == 'openrouter':
                api_key = os.getenv('OPENROUTER_API_KEY')
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY environment variable not set")
                
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    default_headers={
                        "HTTP-Referer": "https://github.com/elizaOS/discord-summarizer",
                        "X-Title": "Discord Q&A Analyzer"
                    }
                )
            
            console.print("[green]✓[/] Model initialized successfully")
        except Exception as e:
            console.print(f"[bold red]Error initializing model:[/] {e}")
            raise

    def _parse_timestamp(self, ts: str) -> str:
        if ts not in self._timestamp_cache:
            self._timestamp_cache[ts] = dateutil.parser.parse(ts).strftime("%H:%M")
        return self._timestamp_cache[ts]

    def _get_user_display_name(self, uid: str) -> str:
        if uid not in self._user_cache:
            user = self.users.get(uid, {})
            self._user_cache[uid] = user.get('nickname') or user.get('name', 'Unknown User')
        return self._user_cache[uid]

    def _chunk_messages(self, messages: List[Dict], chunk_size: int = 15) -> List[List[Dict]]:
        """Simple chunking of messages"""
        return [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]

    def format_messages(self, messages: List[Dict]) -> str:
        """Format messages for analysis"""
        formatted = []
        
        for msg in messages:
            if not msg.get('content', '').strip():
                continue
                
            base_msg = (f"{self._get_user_display_name(msg['uid'])} "
                       f"({self._parse_timestamp(msg['ts'])}): {msg['content']}")
            
            if msg.get('type') == 'Reply' and msg.get('ref'):
                base_msg = f"[Reply to previous message] {base_msg}"
            
            formatted.append(base_msg)
        
        return "\n".join(formatted)

    def analyze_chat(self, chat_data: Dict[str, Any]) -> str:
        """Analyze Discord chat data for Q&A"""
        messages = chat_data.get('messages', [])
        self.users = chat_data.get('users', {})
        
        if not messages:
            return "# No messages to analyze"
        
        chunks = self._chunk_messages(messages)
        analyses = []
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("[cyan]Analyzing chat...", total=len(chunks))
            
            for chunk in chunks:
                try:
                    analysis = self._analyze_chunk(chunk)
                    analyses.append(analysis)
                    progress.advance(task)
                except Exception as e:
                    logger.error(f"Chunk analysis failed: {e}")
                    continue
        
        merged_analysis = self._merge_analyses(analyses)
        return self._format_markdown(merged_analysis, chat_data['channel']['name'], chat_data['date'])

    def _analyze_chunk(self, chunk: List[Dict]) -> ChatAnalysis:
        transcript = self.format_messages(chunk)
        prompt = """Analyze this Discord chat segment and extract meaningful questions and their answers:
    
Focus on:
- Actual questions that received substantive answers
- Technical questions and implementation details
- Include the full context of both question and answer
- Skip rhetorical questions or casual conversation

For each Q&A pair, provide:
1. The exact question as asked
2. Who asked it (use their exact Discord username)
3. The complete answer given
4. Who provided the answer
5. Any important context around the exchange

Ignore:
- Questions without answers
- Simple clarifications
- Repeated questions
- Small talk

Chat transcript:
{transcript}
Return a JSON object with a 'qa_pairs' array containing objects with 'question', 'asker', 'answer', 'answerer', and optional 'context' fields."""

        try:
            if self.model_provider == 'openrouter':
                completion = self.client.chat.completions.create(
                    model=self.openrouter_model,
                    messages=[{
                        "role": "system",
                        "content": "You are a technical assistant that analyzes chat transcripts and outputs JSON."
                    }, {
                        "role": "user",
                        "content": prompt.format(transcript=transcript)
                    }],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                content = completion.choices[0].message.content
            else:    
                response = self.model.invoke(prompt.format(transcript=transcript))
                content = response.content

            # Handle JSON wrapped in markdown code blocks
            json_str = content.strip()
            if json_str.startswith('```json'):
                json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
            elif json_str.startswith('```'):
                json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]

            parsed = json.loads(json_str)
            return ChatAnalysis(qa_pairs=[ChatQA(**qa) for qa in parsed.get('qa_pairs', [])])

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}\nResponse content: {content}")
            return ChatAnalysis()
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return ChatAnalysis()

    def _merge_analyses(self, analyses: List[ChatAnalysis]) -> ChatAnalysis:
        """Merge analyses with deduplication"""
        if not analyses:
            return ChatAnalysis()
    
        seen_qa = set()
        merged = ChatAnalysis()
        
        for analysis in analyses:
            for qa in analysis.qa_pairs:
                key = f"{qa.question}:{qa.asker}"
                if key not in seen_qa:
                    seen_qa.add(key)
                    merged.qa_pairs.append(qa)
        
        return merged

    def _format_markdown(self, analysis: ChatAnalysis, channel_name: str, date: str) -> str:
        """Format Q&A as markdown"""
        sections = [f"# Discord Q&A Summary - {channel_name} {date}\n"]
        
        if analysis.qa_pairs:
            for i, qa in enumerate(analysis.qa_pairs, 1):
                sections.extend([
                    f"## Q{i}: {qa.question}",
                    f"**Asked by:** {qa.asker}",
                    f"\n**Answer:** {qa.answer}",
                    f"**Answered by:** {qa.answerer}"
                ])
                
                if qa.context:
                    sections.append(f"\n**Context:** {qa.context}")
                
                sections.append("\n---\n")
        else:
            sections.append("*No Q&A pairs found in this chat segment*")
        
        return "\n".join(sections)

def main():
    parser = argparse.ArgumentParser(description="Extract Q&A from Discord chat export")
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Path to Discord chat export JSON file")
    parser.add_argument("-o", "--output", type=str,
                       help="Path to save the output file")
    parser.add_argument("--model", type=str, choices=['ollama', 'openrouter'], 
                       default='ollama', help="AI model provider to use")
    parser.add_argument("--openrouter-model", type=str,
                       default="openai/gpt-3.5-turbo",
                       help="OpenRouter model name")

    args = parser.parse_args()

    try:
        chat_data = json.load(open(args.input, 'rb'))
        analyzer = DiscordChatAnalyzer(
            model_provider=args.model,
            openrouter_model=args.openrouter_model
        )
        analysis = analyzer.analyze_chat(chat_data)

        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(analysis)
        else:
            print(analysis)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
