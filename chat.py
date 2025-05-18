import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)
console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Chat interactivo con Phi-2 (CPU)")
    parser.add_argument("--model_path", type=str, default="C:/Users/J Bernardo/Desktop/tp_ia/phi-2-finetuned")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0, -1].item() in self.stop_ids:
            return True
        return False
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )
    return logging.getLogger(__name__)

def main():
    args = parse_args()
    logger = setup_logging()
    device = torch.device("cpu")
    logger.info("🖥️ Ejecutando en CPU")

    tokenizer = AutoTokenizer.from_pretrained("C:/Users/J Bernardo/Desktop/tp_ia/phi-2-finetuned")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.padding_side = "left"
    special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
    tokenizer.save_pretrained(args.model_path)

    model = AutoModelForCausalLM.from_pretrained(
    args.model_path, 
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    ignore_mismatched_sizes=True
)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print("🔢 Tokenizer vocab size:", len(tokenizer))
    print("🔢 Model vocab size:", model.get_input_embeddings().num_embeddings)
    model.eval()
    with torch.no_grad():
        dummy = tokenizer("<|user|> 2+2 <|assistant|>", return_tensors="pt")
        dummy = {k: v.to("cpu") for k, v in dummy.items()}
        out = model(**dummy)
    if torch.isnan(out.logits).any():
        raise RuntimeError("❌ ¡El modelo finetuneado tiene NaNs al primer forward! Posiblemente está dañado.")
    console.print(Panel("[bold green]Modelo listo para chatear![/bold green]\nEscribí 'exit' para salir", title="Phi-2 Chatbot", border_style="blue"))

    history = [
        {"role": "user", "content": "¿Cuánto es 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4"}
]

    try:
        while True:
            user_input = console.input("[bold cyan]Tú:[/bold cyan] ")
            if user_input.strip().lower() in ("exit", "quit"):
                console.print("[bold yellow]Adiós![/bold yellow]")
                break

            history.append({"role": "user", "content": user_input})

            prompt = ""
            for turn in history:
                prompt += turn['content'].strip() + "\n"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
     
            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits         

                stop_token_ids = [tokenizer.eos_token_id]
                stopping_criteria = StoppingCriteriaList([
    StopOnTokens([tokenizer.eos_token_id])
])
            # Generación del modelo
            outputs = model.generate(
                **inputs,
                min_new_tokens=1,
                temperature=0.5,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=64,
                stopping_criteria=stopping_criteria
)
            generated = outputs[0][inputs["input_ids"].shape[-1]:]

            if generated.numel() == 0:
                    console.print("[bold red]⚠️ El modelo no generó ningún token. Saltando turno.[/bold red]")
                    continue  # Saltar al siguiente turno

                # Solo si `generated` tiene contenido, procesar respuesta
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            for stop in ["<|user|>", "<|assistant|>", "<|endoftext|>"]:
                    if stop in answer:
                        answer = answer.split(stop)[0]
            answer = answer.strip(" \n`")

            console.print(f"[bold green]IA:[/bold green] {answer}")
            history.append({"role": "assistant", "content": answer})
            if answer.lower().startswith(user_input.lower()):
                answer = answer[len(user_input):].strip(" :=")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrumpido por el usuario[/bold yellow]")
    finally:
        del model
        gc.collect()
        logger.info("🧹 Recursos liberados.")

if __name__ == "__main__":
    main()
