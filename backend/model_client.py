import os
import re
import sys
from dotenv import load_dotenv
from openai import OpenAI
from models_config import get_model_config
from transformers import AutoModel, AutoTokenizer

def clean_markdown(text: str) -> str:
    # Remove code blocks
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`.*?`', ' ', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove lists
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
    # Remove emphasis
    text = re.sub(r'\*\*|__', ' ', text)
    # Remove links
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    # Remove horizontal rules
    text = re.sub(r'^[-=]{3,}$', ' ', text, flags=re.MULTILINE)
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ModelClientFactory:
    @staticmethod
    def get_client(model_id=None):
        config = get_model_config(model_id)
        
        if config["provider"] == "openai":
            return OpenAIClient(config)
        elif config["provider"] == "huggingface":
            return HuggingFaceClient(config)
        elif config["provider"] == "paperclip":
            return PaperClipClient(config)
        else:
            raise ValueError(f"Unsupported provider: {config['provider']}")

class ModelClient:
    def __init__(self, config):
        self.config = config
    
    def generate_text(self, prompt, **kwargs):
        raise NotImplementedError("Subclasses must implement generate_text")
    
    def generate_summary(self, text, **kwargs):
        raise NotImplementedError("Subclasses must implement generate_summary")
    
    def generate_chat_response(self, query, paper_id, **kwargs):
        raise NotImplementedError("Subclasses must implement generate_chat_response")
    
    def get_context(self, paper_id):
        papers_dir = os.path.join(os.path.dirname(__file__), 'papers')
        paper_path = os.path.join(papers_dir, paper_id, f'{paper_id}.md')
        context_path = os.path.join(papers_dir, paper_id, 'context.txt')
        
        if not os.path.exists(paper_path):
            raise FileNotFoundError(f"Paper not found: {paper_path}")
        
        if not os.path.exists(context_path):
            print("No context path, making new file")
            os.makedirs(os.path.dirname(context_path), exist_ok=True)
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_content = f.read().strip()
            with open(context_path, 'w', encoding='utf-8') as f:
                f.write(paper_content + "\n---------------------------------------------------------------------")

        with open(context_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def save_context(self, paper_id, new_context):
        papers_dir = os.path.join(os.path.dirname(__file__), 'papers')
        context_path = os.path.join(papers_dir, paper_id, 'context.txt')
        
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        with open(context_path, 'w', encoding='utf-8') as f:
            print("Saving context!")
            f.write(new_context)
    
    def format_context(self, existing_context, query, response):
        return f"{existing_context}\n\nQuery: {query}\n\nResponse: {response}"

class OpenAIClient(ModelClient):
    def __init__(self, config):
        load_dotenv()
        super().__init__(config)
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        if not api_key:
            raise ValueError(f"API key environment variable {config['api_key_env']} not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate_text(self, prompt, **kwargs):
        temperature = kwargs.get("temperature", self.config["temperature"])
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        
        response = self.client.completions.create(
            model=self.config["model_id"],
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].text.strip()
    
    def generate_summary(self, text, **kwargs):
        prompt = f"""You are a helpful assistant that summarizes scientific papers. 
        You have a conversational but professional tone, and are trying to synthesize information in the most accessible way possible.
            
        When writing a summary, make sure to add line breaks and formatting to make things possible to read quickly and easily. 
        If using technical terms or abbreviations, give context or a brief explanation.
        Keep your summaries to no more than a few short paragraphs.
        
        Please summarize the following paper in markdown format:
    
        {text}

        Summary:"""
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes scientific papers."},
                {"role": "user", "content": prompt}
            ],
            model=self.config["model_id"],
            temperature=kwargs.get("temperature", self.config["temperature"]),
            max_tokens=kwargs.get("max_tokens", self.config["max_tokens"])
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_chat_response(self, query, paper_id, **kwargs):
        try:
            context = self.get_context(paper_id)
            clean_context = clean_markdown(context)

            SYSTEM_PROMPT = """You are a helpful assistant that concisely summarizes scientific papers. 
Your task is to answer questions about the paper, including giving context to specific parts of the text and extrapolating for the user.
Answer with friendly, conversational language, and make the user feel comfortable talking about anything related to the paper.
If you don't know the answer, say so - don't make things up.
Be brief and to the point!"""
            
            USER_PROMPT = f"""Paper Content:
{clean_context}

User Question: {query}"""
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                model=self.config["model_id"],
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 200)
            )
            
            clean_response = clean_markdown(response.choices[0].message.content)
            
            new_context = self.format_context(context, query, clean_response)
            self.save_context(paper_id, new_context)
            
            return clean_response
            
        except Exception as e:
            print(f"Error in chat function: {str(e)}", file=sys.stderr)
            raise

class HuggingFaceClient(ModelClient):
    def __init__(self, config):
        load_dotenv()
        super().__init__(config)
        
        try:
            import torch
            import gc
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.torch = torch
            self.gc = gc
            self.transformers = True
        except ImportError:
            print("no transformers")
            self.transformers = False
        
        self.model = None
        self.tokenizer = None
        self.generator = None
    
    # memory issues require us to only load models when necessary
    def _load_model_if_needed(self):
        if self.model is not None:
            return True
            
        try:
            if hasattr(self.torch, 'mps') and self.torch.backends.mps.is_available():
                self.device = "mps" 
            else:
                self.device = "cpu"
            print(f"Using device: {self.device} for Hugging Face inference")
            
            self.gc.collect()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            
            print(f"Loading model: {self.config['model_id']}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'])
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_id'],
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory={0: "3GiB", "cpu": "4GiB"}
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print(f"Successfully loaded model: {self.config['model_id']}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.generator
            self.model = None
            self.generator = None
            
            self.gc.collect()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            print("Model unloaded to free memory")
    
    def generate_text(self, prompt, **kwargs):
        if not self.transformers:
            return "Error: Transformers library not available"
        
        if not self._load_model_if_needed():
            return "Error: Failed to load model due to memory constraints"
        
        temperature = kwargs.get("temperature", self.config["temperature"])
        if temperature == 0:
            temperature = 0.2  
        
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        max_tokens = min(max_tokens, 500) 
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )
            
            generated_text = result[0]["generated_text"]
            
            self._unload_model()
                
            return generated_text
        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            self._unload_model()  # need to unload on error as well
            return f"Error generating text: {str(e)}"
    
    def generate_summary(self, text, **kwargs):
        prompt = f"""You are a helpful assistant that summarizes scientific papers. 
        You have a conversational but professional tone, and are trying to synthesize information in the most accessible way possible.
            
        When writing a summary, make sure to add line breaks and formatting to make things possible to read quickly and easily. 
        If using technical terms or abbreviations, give context or a brief explanation.
        Keep your summaries to no more than a few short paragraphs.
        
        Please summarize the following paper in markdown format:
    
        {text[:5000]}

        Summary:"""
        
        return self.generate_text(prompt, **kwargs)
    
    def generate_chat_response(self, query, paper_id, **kwargs):
        try:
            context = self.get_context(paper_id)
            
            clean_context = clean_markdown(context)
            if len(clean_context) > 6000:
                clean_context = clean_context[:6000] + "..."

            prompt = f"""<|system|>
You are a helpful assistant that summarizes scientific papers. 
Your task is to answer questions about the paper, including giving context to specific parts of the text and extrapolating for the user.
Answer with friendly, conversational language, and make the user feel comfortable talking about anything related to the paper.
If you don't know the answer, say so - don't make things up.

<|user|>
Paper Content:
{clean_context}

User Question: {query}

<|assistant|>"""
            
            response = self.generate_text(prompt, **kwargs)
            
            new_context = self.format_context(context, query, response)
            self.save_context(paper_id, new_context)
            
            return response
            
        except Exception as e:
            print(f"Error in chat function: {str(e)}")
            raise

class PaperClipClient(ModelClient):
    def __init__(self, config):
        load_dotenv()
        super().__init__(config)
        
        try:
            import torch
            import gc
            import os
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            self.torch = torch
            self.gc = gc
            self.os = os
            self.peft = PeftModel
            self.transformers = True
        except ImportError:
            print("Warning: Transformers or PEFT not installed.")
            self.transformers = False
        
        self.model = None
        self.tokenizer = None
        
        # Create offload directory if it doesn't exist
        if not os.path.exists("offload_dir"):
            os.makedirs("offload_dir")
    
    def _load_model_if_needed(self):
        if self.model is not None:
            return True
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # Detect device but force CPU for consistency
            self.device = "cpu"  # Force CPU to avoid MPS issues
            print(f"Using device: {self.device} for PaperClip inference (forced to CPU for stability)")
            
            # Aggressive garbage collection
            self.gc.collect()
            if hasattr(self.torch, 'cuda') and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            
            # Load tokenizer first
            print(f"Loading tokenizer for: {self.config['base_model_id']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['base_model_id'],
                use_fast=False
            )
            
            # Load base model
            print(f"Loading base model: {self.config['base_model_id']}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config['base_model_id'],
                device_map={"": self.device},  # Explicitly map all to CPU
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                offload_folder="offload_dir"
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load PEFT model
            print(f"Loading PEFT model: {self.config['model_id']}")
            
            try:
                # Try direct loading first
                self.model = PeftModel.from_pretrained(base_model, self.config['model_id'])
                self.model.to(self.device)  # Ensure model is on CPU
            except Exception as e:
                print(f"Direct loading failed: {e}")
                print("Falling back to base model only")
                self.model = base_model  # Just use base model if PEFT fails
                self.model.to(self.device)  # Ensure model is on CPU
            
            self.model.eval()  # Set to evaluation mode
            print("Model loaded successfully on CPU")
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def _unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            self.gc.collect()
            if hasattr(self.torch, 'cuda') and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            print("Model unloaded")
    
    def generate_text(self, prompt, **kwargs):
        if not self.transformers:
            return "Error: Transformers or PEFT libraries not available"
        
        if not self._load_model_if_needed():
            return "Error: Failed to load model"
        
        try:
            import torch
            
            # Ensure we're using CPU consistently
            self.model = self.model.to("cpu")
            
            # Tokenize input with padding and truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            )
            
            # Ensure inputs are on CPU
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            with torch.no_grad():
                # Generate with attention to device placement
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.2,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Get just the generated part (after prompt)
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            # Clean up repetitive patterns
            if "Contents\nReferences" in response:
                response = response[:response.find("Contents\nReferences")].strip()
            
            self._unload_model()
            return response
            
        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            self._unload_model()
            return f"Error generating text: {str(e)}"
    
    def generate_summary(self, text, **kwargs):
        # Use simple prompt format based on the benchmark
        prompt = f"# Text\n\n{text[:5000]}\n\n# Summary\n\n"
        
        return self.generate_text(prompt, **kwargs)
    
    def generate_chat_response(self, query, paper_id, **kwargs):
        try:
            context = self.get_context(paper_id)
            clean_context = clean_markdown(context)
            if len(clean_context) > 5000:
                clean_context = clean_context[:5000] + "..."
                
            prompt = f"# Paper\n\n{clean_context}\n\n# Question\n\n{query}\n\n# Answer\n\n"
            
            response = self.generate_text(prompt, **kwargs)
            
            new_context = self.format_context(context, query, response)
            self.save_context(paper_id, new_context)
            
            return response
            
        except Exception as e:
            print(f"Error in chat function: {str(e)}")
            raise