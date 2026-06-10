export const openAIChatCompletionFixture = {
    id: 'chatcmpl-fixture-success',
    object: 'chat.completion',
    created: 1700000000,
    model: 'gpt-4o-mini-2024-07-18',
    choices: [{
        index: 0,
        message: {
            role: 'assistant',
            content: 'fixture ok',
            refusal: null,
            annotations: [],
        },
        logprobs: null,
        finish_reason: 'stop',
    }],
    usage: {
        prompt_tokens: 13,
        completion_tokens: 2,
        total_tokens: 15,
        prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
        completion_tokens_details: {
            reasoning_tokens: 0,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        },
    },
    service_tier: 'default',
    system_fingerprint: 'fp_fixture_success',
} as const

export const openAIChatToolCallFixture = {
    id: 'chatcmpl-fixture-tool',
    object: 'chat.completion',
    created: 1700000000,
    model: 'gpt-4o-mini-2024-07-18',
    choices: [{
        index: 0,
        message: {
            role: 'assistant',
            content: null,
            tool_calls: [{
                id: 'call_fixture_lookup',
                type: 'function',
                function: { name: 'fixture_lookup', arguments: '{"query":"abc"}' },
            }],
            refusal: null,
            annotations: [],
        },
        logprobs: null,
        finish_reason: 'stop',
    }],
    usage: {
        prompt_tokens: 18,
        completion_tokens: 7,
        total_tokens: 25,
        prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
        completion_tokens_details: {
            reasoning_tokens: 0,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        },
    },
    service_tier: 'default',
    system_fingerprint: 'fp_fixture_tool',
} as const

export const openAIChatStreamingEventsFixture = [
    'data: {"id":"chatcmpl-fixture-stream","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini-2024-07-18","service_tier":"default","system_fingerprint":"fp_fixture_stream","choices":[{"index":0,"delta":{"role":"assistant","content":"","refusal":null},"logprobs":null,"finish_reason":null}],"obfuscation":"fixture_a"}\n\n',
    'data: {"id":"chatcmpl-fixture-stream","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini-2024-07-18","service_tier":"default","system_fingerprint":"fp_fixture_stream","choices":[{"index":0,"delta":{"content":"stream"},"logprobs":null,"finish_reason":null}],"obfuscation":"fixture_b"}\n\n',
    'data: {"id":"chatcmpl-fixture-stream","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini-2024-07-18","service_tier":"default","system_fingerprint":"fp_fixture_stream","choices":[{"index":0,"delta":{"content":" ok"},"logprobs":null,"finish_reason":null}],"obfuscation":"fixture_c"}\n\n',
    'data: {"id":"chatcmpl-fixture-stream","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini-2024-07-18","service_tier":"default","system_fingerprint":"fp_fixture_stream","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}],"obfuscation":"fixture_d"}\n\n',
    'data: [DONE]\n\n',
] as const

export const openAIChatErrorFixture = {
    error: {
        message: 'The model `not-a-real-openai-model-for-fixtures` does not exist or you do not have access to it.',
        type: 'invalid_request_error',
        param: null,
        code: 'model_not_found',
    },
} as const

export const mistralChatCompletionFixture = {
    id: 'mistral-fixture-success',
    object: 'chat.completion',
    created: 1700000000,
    model: 'mistral-small-latest',
    usage: {
        prompt_tokens: 22,
        total_tokens: 27,
        completion_tokens: 5,
        prompt_tokens_details: { cached_tokens: 0 },
    },
    choices: [{
        index: 0,
        finish_reason: 'stop',
        message: {
            role: 'assistant',
            tool_calls: null,
            content: 'mistral fixture ok',
        },
    }],
} as const

export const mistralChatErrorFixture = {
    object: 'error',
    message: 'Invalid model: not-a-real-mistral-model-for-fixtures',
    type: 'invalid_model',
    param: null,
    code: '1500',
    raw_status_code: 400,
} as const

export const deepSeekReasonerFixture = {
    id: 'deepseek-fixture-reasoner',
    object: 'chat.completion',
    created: 1700000000,
    model: 'deepseek-v4-flash',
    choices: [{
        index: 0,
        message: {
            role: 'assistant',
            content: 'OK',
            reasoning_content: 'We need to answer with only the word OK after reasoning briefly. The instruction is to reason briefly and then output "OK". No other text. So I\'ll just think and then say OK.',
        },
        logprobs: null,
        finish_reason: 'stop',
    }],
    usage: {
        prompt_tokens: 14,
        completion_tokens: 41,
        total_tokens: 55,
        prompt_tokens_details: { cached_tokens: 0 },
        completion_tokens_details: { reasoning_tokens: 39 },
        prompt_cache_hit_tokens: 0,
        prompt_cache_miss_tokens: 14,
    },
    system_fingerprint: 'fp_fixture_deepseek',
} as const

export const openRouterReasoningFixture = {
    id: 'openrouter-fixture-reasoning',
    object: 'chat.completion',
    created: 1700000000,
    model: 'deepseek/deepseek-r1-0528',
    provider: 'AtlasCloud',
    system_fingerprint: null,
    service_tier: null,
    choices: [{
        index: 0,
        logprobs: null,
        message: {
            role: 'assistant',
            content: 'OK',
            refusal: null,
            reasoning: 'Hmm, the user wants me to answer with only "OK" after some brief reasoning.',
            reasoning_details: [{
                type: 'reasoning.text',
                text: 'Hmm, the user wants me to answer with only "OK" after some brief reasoning.',
                format: 'unknown',
                index: 0,
            }],
        },
        finish_reason: 'stop',
        native_finish_reason: 'stop',
    }],
    usage: {
        prompt_tokens: 13,
        completion_tokens: 201,
        total_tokens: 214,
        is_byok: false,
        prompt_tokens_details: {
            cached_tokens: 0,
            cache_write_tokens: 0,
            audio_tokens: 0,
            video_tokens: 0,
        },
        completion_tokens_details: {
            reasoning_tokens: 199,
            image_tokens: 0,
            audio_tokens: 0,
        },
    },
} as const
