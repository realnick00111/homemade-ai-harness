import argparse
import time

import ollama


def parse_context_sizes(value: str) -> list[int]:
    try:
        sizes = [int(size.strip()) for size in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Context sizes must be comma-separated integers.') from exc

    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError('Context sizes must all be positive integers.')
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError('Context sizes must not contain duplicates.')
    return sizes


def benchmark(model: str, num_predict: int, context_size: int = 4096) -> None:
    messages = [
        {
            'role': 'user',
            'content': (
                'Explain how a small community garden can benefit a neighborhood. '
                'Write a detailed answer in several paragraphs.'
            ),
        },
    ]

    print('--- Benchmark Response ---')
    started = time.perf_counter()
    last_chunk = None

    for chunk in ollama.chat(
        model=model,
        messages=messages,
        options={'num_predict': num_predict, 'num_ctx': context_size},
        stream=True,
        think=False,
    ):
        last_chunk = chunk
        if chunk.message.content:
            print(chunk.message.content, end='', flush=True)

    wall_seconds = time.perf_counter() - started
    print('\n--- Benchmark Results ---')

    if last_chunk is None or last_chunk.eval_count is None or last_chunk.eval_duration is None:
        raise RuntimeError('Ollama did not return token count and generation duration metrics.')
    if last_chunk.eval_duration <= 0:
        raise RuntimeError(f'Ollama returned an invalid generation duration: {last_chunk.eval_duration}')

    tokens_per_second = last_chunk.eval_count / (last_chunk.eval_duration / 1_000_000_000)
    print(f'Model: {model}')
    print(f'Configured context window: {context_size} tokens')
    if last_chunk.prompt_eval_count is not None:
        print(f'Prompt tokens processed: {last_chunk.prompt_eval_count}')
    if last_chunk.prompt_eval_duration is not None and last_chunk.prompt_eval_duration > 0:
        prompt_tokens_per_second = (
            last_chunk.prompt_eval_count / (last_chunk.prompt_eval_duration / 1_000_000_000)
            if last_chunk.prompt_eval_count is not None
            else None
        )
        print(f'Prompt processing time: {last_chunk.prompt_eval_duration / 1_000_000_000:.2f} seconds')
        if prompt_tokens_per_second is not None:
            print(f'Prompt processing speed: {prompt_tokens_per_second:.2f} tokens/second')
    print(f'Generated tokens: {last_chunk.eval_count}')
    print(f'Model generation time: {last_chunk.eval_duration / 1_000_000_000:.2f} seconds')
    print(f'Model generation speed: {tokens_per_second:.2f} tokens/second')
    print(f'End-to-end time: {wall_seconds:.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure Ollama model generation speed.')
    parser.add_argument('--model', default='qwen3.5:9b', help='Ollama model to benchmark')
    parser.add_argument(
        '--num-predict',
        type=int,
        default=256,
        help='Maximum number of generated tokens',
    )
    parser.add_argument(
        '--context-sizes',
        type=parse_context_sizes,
        default=parse_context_sizes('32768, 65536, 131072, 262144'),
        metavar='TOKENS[,TOKENS...]',
        help='Comma-separated Ollama context window sizes to benchmark (default: 32768, 65536, 131072, 262144)',
    )
    args = parser.parse_args()
    for context_size in args.context_sizes:
        benchmark(args.model, args.num_predict, context_size)
