import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { LLMFlags, LLMFormat, LLMProvider, LLMTokenizer } from 'src/ts/model/types'
import { fetchNative, textifyReadableStream } from 'src/ts/globalApi.svelte'
import { alertError } from 'src/ts/alert'
import { callTool, decodeToolCall, encodeToolCall } from '../../mcp/mcp'
import { deepSeekReasonerFixture, mistralChatCompletionFixture, mistralChatErrorFixture, openAIChatCompletionFixture, openAIChatErrorFixture, openAIChatStreamingEventsFixture, openAIChatToolCallFixture, openRouterReasoningFixture } from './fixtures/chatCompletions'
import { requestOpenAI } from './requests'

const mocks = vi.hoisted(() => ({
    db: {
        OAIPrediction: '',
        OaiCompAPIKeys: {},
        PresensePenalty: 11,
        additionalParams: [],
        autofillRequestUrl: false,
        cipherChat: true,
        customModels: [],
        customProxyRequestModel: 'custom-proxy-model',
        deepseekReasoningEffort: 'high',
        deepseekThinkingType: 'disabled',
        frequencyPenalty: 22,
        genTime: 2,
        generationSeed: 0,
        gptVisionQuality: 'high',
        jsonSchema: '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}',
        jsonSchemaEnabled: false,
        localNetworkMode: false,
        localNetworkTimeoutSec: 600,
        modelTools: [] as string[],
        nanogptKey: 'nanogpt-key',
        nanogptProvider: '',
        nanogptRequestModel: 'nanogpt-model',
        nanogptUseSubscriptionEndpoint: false,
        newOAIHandle: true,
        openAIKey: 'openai-key',
        openAIFlexProcessing: false,
        openrouterFallback: false,
        openrouterKey: 'openrouter-key',
        openrouterMiddleOut: false,
        openrouterProvider: undefined as any,
        openrouterRequestModel: 'openrouter/model',
        proxyKey: 'proxy-key',
        proxyRequestModel: 'proxy-model',
        requestRetrys: 0,
        reverseProxyOobaArgs: {},
        reverseProxyOobaMode: false,
        reasoningEffort: 2,
        seperateParameters: {},
        seperateParametersByModel: false,
        seperateParametersEnabled: false,
        simplifiedToolUse: false,
        strictJsonSchema: true,
        temperature: 70,
        thinkingTokens: 128,
        top_k: 0,
        top_p: 0.9,
        top_a: 0,
        min_p: 0,
        repetition_penalty: 1.05,
        useInstructPrompt: false,
        verbosity: 0,
    },
    fetchNative: vi.fn(),
    getFreeOpenRouterModels: vi.fn(),
    globalFetch: vi.fn(),
    platform: {
        isNodeServer: true,
        isTauri: false,
    },
    supportsInlayImage: vi.fn(() => false),
}))

vi.mock('src/ts/storage/database.svelte', () => ({
    getDatabase: () => mocks.db,
}))

vi.mock('src/ts/globalApi.svelte', () => ({
    addFetchLog: vi.fn(),
    fetchNative: mocks.fetchNative,
    globalFetch: mocks.globalFetch,
    textifyReadableStream: vi.fn(async () => 'stream error body'),
}))

vi.mock('src/lang', () => ({
    language: { errors: { httpError: 'HTTP ' } },
}))

vi.mock('src/ts/alert', () => ({
    alertError: vi.fn(),
}))

vi.mock('src/ts/platform', () => ({
    get isNodeServer() {
        return mocks.platform.isNodeServer
    },
    get isTauri() {
        return mocks.platform.isTauri
    },
}))

vi.mock('../../templates/jsonSchema', () => ({
    extractJSON: (data: string) => data,
    getOpenAIJSONSchema: () => ({
        name: 'format',
        strict: true,
        schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'], additionalProperties: false },
    }),
}))

vi.mock('src/ts/process/templates/jsonSchema', () => ({
    extractJSON: (data: string) => data,
    getOpenAIJSONSchema: () => ({
        name: 'format',
        strict: true,
        schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'], additionalProperties: false },
    }),
}))

vi.mock('../../templates/chatTemplate', () => ({
    applyChatTemplate: vi.fn(() => 'templated prompt'),
}))

vi.mock('src/ts/process/templates/chatTemplate', () => ({
    applyChatTemplate: vi.fn(() => 'templated prompt'),
}))

vi.mock('src/ts/model/modellist', () => ({
    LLMFlags: {
        DeveloperRole: 14,
        OAICompletionTokens: 13,
        deepSeekPrefix: 17,
        deepSeekThinkingInput: 18,
        deepSeekThinkingOutput: 19,
        deepSeekThinkingToggle: 24,
        noStructuredOutput: 25,
    },
    LLMFormat: {
        Mistral: 4,
        OpenAICompatible: 0,
    },
    LLMProvider: {
        OpenAI: 0,
        AsIs: 4,
    },
}))

vi.mock('src/ts/tokenizer', () => ({
    strongBan: vi.fn(async (_text: string, bias: Record<string, number>) => ({ ...bias, '999': -100 })),
    tokenizeNum: vi.fn(async () => [123, 456]),
}))

vi.mock('src/ts/model/openrouter', () => ({
    getFreeOpenRouterModels: mocks.getFreeOpenRouterModels,
}))

vi.mock('src/ts/util', () => ({
    simplifySchema: (schema: unknown) => schema,
}))

vi.mock('../../files/inlays', () => ({
    supportsInlayImage: mocks.supportsInlayImage,
}))

vi.mock('../../mcp/mcp', () => ({
    callTool: vi.fn(),
    decodeToolCall: vi.fn(),
    encodeToolCall: vi.fn(),
}))

const baseArg = (overrides: Record<string, any> = {}) => ({
    aiModel: 'gpt-5-chat',
    bias: {},
    biasString: [],
    formated: [
        { role: 'system', content: 'Follow policy.', memo: 'drop', removable: true, attr: { hidden: true } },
        { role: 'user', content: 'Hello' },
    ],
    maxTokens: 321,
    mode: 'model',
    modelInfo: {
        flags: [LLMFlags.DeveloperRole],
        format: LLMFormat.OpenAICompatible,
        id: 'gpt-5-chat',
        internalID: 'gpt-5',
        name: 'GPT-5 Chat',
        parameters: ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'reasoning_effort', 'verbosity'],
        provider: LLMProvider.OpenAI,
        tokenizer: LLMTokenizer.Unknown,
    },
    ...overrides,
}) as any

async function collectStream(stream: ReadableStream<Record<string, string>>) {
    const reader = stream.getReader()
    const chunks: Record<string, string>[] = []
    while(true){
        const { done, value } = await reader.read()
        if(done){
            return chunks
        }
        chunks.push(value)
    }
}

function sseStream(events: string[]) {
    const encoder = new TextEncoder()
    return new ReadableStream<Uint8Array>({
        start(controller) {
            for(const event of events){
                controller.enqueue(encoder.encode(event))
            }
            controller.close()
        }
    })
}

function streamResponse(events: string[], status = 200, contentType = 'text/event-stream') {
    return {
        status,
        headers: { get: () => contentType },
        body: sseStream(events),
    }
}

describe('OpenAI Chat Completions requests', () => {
    let consoleLogSpy: ReturnType<typeof vi.spyOn>

    beforeAll(() => {
        vi.stubGlobal('safeStructuredClone', (value: unknown) => structuredClone(value))
        consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    })

    afterAll(() => {
        vi.unstubAllGlobals()
        consoleLogSpy.mockRestore()
    })

    beforeEach(() => {
        mocks.fetchNative.mockReset()
        mocks.getFreeOpenRouterModels.mockReset()
        mocks.globalFetch.mockReset()
        mocks.supportsInlayImage.mockReset()
        mocks.supportsInlayImage.mockReturnValue(false)
        mocks.db.OAIPrediction = ''
        mocks.db.OaiCompAPIKeys = {}
        mocks.db.PresensePenalty = 11
        mocks.db.additionalParams = []
        mocks.db.autofillRequestUrl = false
        mocks.db.cipherChat = true
        mocks.db.customModels = []
        mocks.db.customProxyRequestModel = 'custom-proxy-model'
        mocks.db.deepseekReasoningEffort = 'high'
        mocks.db.deepseekThinkingType = 'disabled'
        mocks.db.frequencyPenalty = 22
        mocks.db.genTime = 2
        mocks.db.generationSeed = 0
        mocks.db.gptVisionQuality = 'high'
        mocks.db.jsonSchemaEnabled = false
        mocks.db.localNetworkMode = false
        mocks.db.localNetworkTimeoutSec = 600
        mocks.db.modelTools = []
        mocks.db.nanogptProvider = ''
        mocks.db.nanogptRequestModel = 'nanogpt-model'
        mocks.db.nanogptUseSubscriptionEndpoint = false
        mocks.db.newOAIHandle = true
        mocks.db.openAIFlexProcessing = false
        mocks.db.openrouterFallback = false
        mocks.db.openrouterMiddleOut = false
        mocks.db.openrouterProvider = undefined
        mocks.db.openrouterRequestModel = 'openrouter/model'
        mocks.db.requestRetrys = 0
        mocks.db.reverseProxyOobaArgs = {}
        mocks.db.reverseProxyOobaMode = false
        mocks.db.reasoningEffort = 2
        mocks.db.seperateParametersEnabled = false
        mocks.db.simplifiedToolUse = false
        mocks.db.temperature = 70
        mocks.db.top_p = 0.9
        mocks.db.useInstructPrompt = false
        mocks.db.verbosity = 0
        mocks.platform.isNodeServer = true
        mocks.platform.isTauri = false
        vi.mocked(textifyReadableStream).mockClear()
        vi.mocked(alertError).mockReset()
        vi.mocked(callTool).mockReset()
        vi.mocked(decodeToolCall).mockReset()
        vi.mocked(encodeToolCall).mockReset()
    })

    it('builds official OpenAI defaults with bearer auth, internal model id, messages, and model parameters', async () => {
        const result = await requestOpenAI(baseArg({ previewBody: true }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://api.openai.com/v1/chat/completions')
        expect(preview.headers.Authorization).toBe('Bearer openai-key')
        expect(preview.headers['Content-Type']).toBe('application/json')
        expect(preview.body).toMatchObject({
            model: 'gpt-5',
            max_tokens: 321,
            stream: false,
            temperature: 0.7,
            top_p: 0.9,
            frequency_penalty: 0.22,
            presence_penalty: 0.11,
            reasoning_effort: 'high',
            verbosity: 'low',
        })
        expect(preview.body.messages).toEqual([
            { role: 'developer', content: 'Follow policy.', name: undefined },
            { role: 'user', content: 'Hello', name: undefined },
        ])
    })

    it('uses max_completion_tokens for models flagged with OAICompletionTokens', async () => {
        const result = await requestOpenAI(baseArg({
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [LLMFlags.OAICompletionTokens],
            },
        }))

        expect(result.type).toBe('success')
        const body = JSON.parse(result.result as string).body
        expect(body.max_completion_tokens).toBe(321)
        expect(body.max_tokens).toBeUndefined()
    })

    it('applies bias strings, token ids, strong bans, seeds, and prediction content', async () => {
        mocks.db.generationSeed = 42
        mocks.db.OAIPrediction = 'predicted continuation'

        const result = await requestOpenAI(baseArg({
            previewBody: true,
            bias: { 7: 1 },
            biasString: [
                ['[[42]]', -5],
                ['hard ban', -101],
                ['soft bias', 3],
            ],
        }))

        expect(result.type).toBe('success')
        const body = JSON.parse(result.result as string).body
        expect(body.logit_bias).toEqual({
            '7': 1,
            '42': -5,
            '123': 3,
            '456': 3,
            '999': -100,
        })
        expect(body.seed).toBe(42)
        expect(body.prediction).toEqual({ type: 'content', content: 'predicted continuation' })
    })

    it('leaves system role unchanged without the developer-role flag and strips internal message metadata', async () => {
        const result = await requestOpenAI(baseArg({
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
            },
        }))

        expect(result.type).toBe('success')
        const messages = JSON.parse(result.result as string).body.messages
        expect(messages[0]).toEqual({ role: 'system', content: 'Follow policy.', name: undefined })
        expect(messages[0]).not.toHaveProperty('memo')
        expect(messages[0]).not.toHaveProperty('removable')
        expect(messages[0]).not.toHaveProperty('attr')
    })

    it('converts multimodal user messages to OpenAI image_url content blocks without mutating the source', async () => {
        const sourceMessages = [
            { role: 'user', content: 'Look', multimodals: [{ type: 'image', base64: 'data:image/png;base64,abc' }] },
        ]

        const result = await requestOpenAI(baseArg({ formated: sourceMessages, previewBody: true }))

        expect(result.type).toBe('success')
        const messages = JSON.parse(result.result as string).body.messages
        expect(messages[0].content).toEqual([
            { type: 'image_url', image_url: { url: 'data:image/png;base64,abc', detail: 'high' } },
            { type: 'text', text: 'Look' },
        ])
        expect(sourceMessages[0]).toEqual({ role: 'user', content: 'Look', multimodals: [{ type: 'image', base64: 'data:image/png;base64,abc' }] })
    })

    it('filters empty new-OAI messages while keeping tool messages', async () => {
        const result = await requestOpenAI(baseArg({
            previewBody: true,
            formated: [
                { role: 'user', content: '', memo: 'NewChat: cleared' },
                { role: 'tool', content: '', tool_call_id: 'call_1' },
                { role: 'assistant', content: 'Kept' },
            ],
        }))

        expect(result.type).toBe('success')
        const messages = JSON.parse(result.result as string).body.messages
        expect(messages).toEqual([
            { role: 'tool', content: '', tool_call_id: 'call_1' },
            { role: 'assistant', content: 'Kept', name: undefined },
        ])
    })

    it('decodes remembered tool_call markup into assistant and tool messages', async () => {
        vi.mocked(decodeToolCall).mockResolvedValueOnce({
            call: { id: 'call_remembered', name: 'lookup', arg: '{"q":"old"}' },
            response: [{ type: 'text', text: 'remembered result' }],
        } as any)

        const result = await requestOpenAI(baseArg({
            previewBody: true,
            formated: [{ role: 'assistant', content: 'Before <tool_call>encoded</tool_call> After' }],
        }))

        expect(result.type).toBe('success')
        const messages = JSON.parse(result.result as string).body.messages
        expect(messages).toEqual([
            {
                role: 'assistant',
                content: 'Before ',
                tool_calls: [{ id: 'call_remembered', type: 'function', function: { name: 'lookup', arguments: '{"q":"old"}' } }],
                name: undefined,
            },
            { role: 'tool', content: 'remembered result', tool_call_id: 'call_remembered' },
            { role: 'assistant', content: ' After', name: undefined },
        ])
    })

    it('applies structured output for schemas unless noStructuredOutput is set', async () => {
        const enabled = await requestOpenAI(baseArg({ previewBody: true, schema: '{"type":"object"}' }))
        expect(enabled.type).toBe('success')
        expect(JSON.parse(enabled.result as string).body.response_format).toEqual({
            type: 'json_schema',
            json_schema: {
                name: 'format',
                strict: true,
                schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'], additionalProperties: false },
            },
        })

        const disabled = await requestOpenAI(baseArg({
            previewBody: true,
            schema: '{"type":"object"}',
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [LLMFlags.noStructuredOutput],
            },
        }))
        expect(disabled.type).toBe('success')
        expect(JSON.parse(disabled.result as string).body.response_format).toBeUndefined()
    })

    it('enables DeepSeek thinking mode and removes sampling controls', async () => {
        mocks.db.deepseekThinkingType = 'enabled'
        mocks.db.deepseekReasoningEffort = 'medium'

        const result = await requestOpenAI(baseArg({
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [LLMFlags.deepSeekThinkingToggle],
            },
        }))

        expect(result.type).toBe('success')
        const body = JSON.parse(result.result as string).body
        expect(body.thinking).toEqual({ type: 'enabled', reasoning_effort: 'medium' })
        expect(body.temperature).toBeUndefined()
        expect(body.top_p).toBeUndefined()
        expect(body.frequency_penalty).toBeUndefined()
        expect(body.presence_penalty).toBeUndefined()
    })

    it('passes DeepSeek assistant prefix and prior reasoning content for supported models', async () => {
        const result = await requestOpenAI(baseArg({
            previewBody: true,
            formated: [
                { role: 'user', content: 'Continue.' },
                { role: 'assistant', content: 'Prefill', thoughts: ['reason one', 'reason two'] },
            ],
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [LLMFlags.deepSeekPrefix, LLMFlags.deepSeekThinkingInput],
            },
        }))

        expect(result.type).toBe('success')
        const messages = JSON.parse(result.result as string).body.messages
        expect(messages.at(-1)).toMatchObject({
            role: 'assistant',
            content: 'Prefill',
            prefix: true,
            reasoning_content: 'reason one\nreason two',
        })
        expect(messages.at(-1)).not.toHaveProperty('thoughts')
    })

    it('disables DeepSeek thinking mode while preserving normal sampling controls', async () => {
        mocks.db.deepseekThinkingType = 'disabled'

        const result = await requestOpenAI(baseArg({
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [LLMFlags.deepSeekThinkingToggle],
            },
        }))

        expect(result.type).toBe('success')
        const body = JSON.parse(result.result as string).body
        expect(body.thinking).toEqual({ type: 'disabled' })
        expect(body.temperature).toBe(0.7)
        expect(body.top_p).toBe(0.9)
        expect(body.frequency_penalty).toBe(0.22)
        expect(body.presence_penalty).toBe(0.11)
    })

    it('applies global additional params to body and headers, including nested deletion and non-stream forcing', async () => {
        mocks.db.additionalParams = [
            ['metadata.source', 'global'],
            ['response_format.json_schema.strict', '{{none}}'],
            ['header::X-Test', 'ok'],
            ['stream', 'true'],
        ]

        const result = await requestOpenAI(baseArg({ previewBody: true, schema: '{"type":"object"}' }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.headers['X-Test']).toBe('ok')
        expect(preview.body.metadata.source).toBe('global')
        expect(preview.body.response_format.json_schema).toEqual({
            name: 'format',
            schema: { type: 'object', properties: { answer: { type: 'string' } }, required: ['answer'], additionalProperties: false },
        })
        expect(preview.body.stream).toBe(false)
    })

    it('applies custom model endpoint, keyIdentifier, and model-specific additional params', async () => {
        mocks.db.OaiCompAPIKeys = { customKey: 'custom-key' }
        mocks.db.additionalParams = [['metadata.global', 'yes']]
        mocks.db.customModels = [{
            id: 'xcustom:::chat',
            params: 'header::X-Custom=yes\nmetadata.tier=gold\nextra=json::{"enabled":true}',
        }]

        const result = await requestOpenAI(baseArg({
            aiModel: 'xcustom:::chat',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                endpoint: 'https://custom.example/v1/chat/completions',
                keyIdentifier: 'customKey',
                internalID: 'custom-internal',
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://custom.example/v1/chat/completions')
        expect(preview.headers.Authorization).toBe('Bearer custom-key')
        expect(preview.headers['X-Custom']).toBe('yes')
        expect(preview.body.model).toBe('custom-internal')
        expect(preview.body.metadata).toEqual({ global: 'yes', tier: 'gold' })
        expect(preview.body.extra).toEqual({ enabled: true })
    })

    it('autofills reverse proxy chat URLs and preserves risu proxy identify headers', async () => {
        mocks.db.autofillRequestUrl = true

        const result = await requestOpenAI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'risu::https://proxy.example/api',
            previewBody: true,
            key: undefined,
            modelInfo: {
                ...baseArg().modelInfo,
                provider: LLMProvider.AsIs,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://proxy.example/api/v1/chat/completions')
        expect(preview.headers.Authorization).toBe('Bearer proxy-key')
        expect(preview.headers['X-Proxy-Risu']).toBe('RisuAI')
    })

    it.each([
        ['https://proxy.example/v1', 'https://proxy.example/v1/chat/completions'],
        ['https://proxy.example/v1/', 'https://proxy.example/v1/chat/completions'],
        ['https://proxy.example/', 'https://proxy.example/v1/chat/completions'],
        ['https://proxy.example/v1/chat/completions', 'https://proxy.example/v1/chat/completions'],
    ])('autofills reverse proxy chat URL variant %s', async (customURL, expectedURL) => {
        mocks.db.autofillRequestUrl = true

        const result = await requestOpenAI(baseArg({
            aiModel: 'reverse_proxy',
            customURL,
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                provider: LLMProvider.AsIs,
            },
        }))

        expect(result.type).toBe('success')
        expect(JSON.parse(result.result as string).url).toBe(expectedURL)
    })

    it('relocates reverse-proxy Ooba system prompts and overlays Ooba args', async () => {
        mocks.db.reverseProxyOobaMode = true
        mocks.db.reverseProxyOobaArgs = { mode: 'chat-instruct', custom_flag: true, ignored_null: null }

        const result = await requestOpenAI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'https://proxy.example/v1/chat/completions',
            previewBody: true,
            formated: [
                { role: 'system', content: 'First system' },
                { role: 'user', content: 'Question' },
                { role: 'system', content: 'Second system' },
            ],
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
                provider: LLMProvider.AsIs,
            },
        }))

        expect(result.type).toBe('success')
        const body = JSON.parse(result.result as string).body
        expect(body.messages).toEqual([
            { role: 'user', content: 'Question', name: undefined },
            { role: 'system', content: 'First system\nSecond system' },
        ])
        expect(body.mode).toBe('chat-instruct')
        expect(body.custom_flag).toBe(true)
        expect(body).not.toHaveProperty('ignored_null')
    })

    it('formats Mistral chat payloads with merged system and function content', async () => {
        const result = await requestOpenAI(baseArg({
            key: 'mistral-key',
            previewBody: true,
            formated: [
                { role: 'system', content: 'System A' },
                { role: 'system', content: 'System B' },
                { role: 'function', content: 'Function output' },
                { role: 'user', content: 'Ask' },
            ],
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
                format: LLMFormat.Mistral,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://api.mistral.ai/v1/chat/completions')
        expect(preview.headers.Authorization).toBe('Bearer mistral-key')
        expect(preview.body).toMatchObject({
            model: 'gpt-5-chat',
            safe_prompt: false,
            max_tokens: 321,
        })
        expect(preview.body.messages).toEqual([
            { role: 'system', content: 'System A\nSystem B' },
            { role: 'user', content: 'Function output\nAsk' },
        ])
    })

    it('formats Mistral alternate role transitions', async () => {
        const result = await requestOpenAI(baseArg({
            previewBody: true,
            formated: [
                { role: 'assistant', content: 'Assistant starts' },
                { role: 'system', content: 'System after assistant' },
                { role: 'assistant', content: 'Assistant continues' },
            ],
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
                format: LLMFormat.Mistral,
            },
        }))

        expect(result.type).toBe('success')
        const messages = JSON.parse(result.result as string).body.messages
        expect(messages).toEqual([
            { role: 'system', content: 'assistant:Assistant starts\nSystem after assistant' },
            { role: 'assistant', content: 'Assistant continues' },
        ])
    })

    it('handles real-shaped Mistral runtime success and error responses', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: structuredClone(mistralChatCompletionFixture),
        })

        const success = await requestOpenAI(baseArg({
            key: 'mistral-key',
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
                format: LLMFormat.Mistral,
            },
        }))

        expect(success).toEqual({ type: 'success', result: 'mistral fixture ok' })
        expect(mocks.globalFetch).toHaveBeenCalledWith('https://api.mistral.ai/v1/chat/completions', expect.objectContaining({
            headers: { Authorization: 'Bearer mistral-key' },
            interceptor: 'mistral',
        }))

        mocks.globalFetch.mockResolvedValueOnce({ ok: true, data: { choices: [] } })
        const malformed = await requestOpenAI(baseArg({
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
                format: LLMFormat.Mistral,
            },
        }))
        expect(malformed).toEqual({ type: 'fail', result: 'HTTP {"choices":[]}' })

        mocks.globalFetch.mockResolvedValueOnce({ ok: false, data: structuredClone(mistralChatErrorFixture) })
        const providerError = await requestOpenAI(baseArg({
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
                format: LLMFormat.Mistral,
            },
        }))
        expect(providerError).toEqual({ type: 'fail', result: 'HTTP {"object":"error","message":"Invalid model: not-a-real-mistral-model-for-fixtures","type":"invalid_model","param":null,"code":"1500","raw_status_code":400}' })
    })

    it('builds OpenRouter headers, free model routing, transforms, route, and provider settings', async () => {
        mocks.db.openrouterRequestModel = 'risu/free'
        mocks.db.openrouterFallback = true
        mocks.db.openrouterMiddleOut = true
        mocks.db.openrouterProvider = {
            order: ['Anthropic'],
            only: ['OpenAI'],
            ignore: [],
        }
        mocks.getFreeOpenRouterModels.mockResolvedValueOnce('free/model')

        const result = await requestOpenAI(baseArg({
            aiModel: 'openrouter',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                provider: LLMProvider.AsIs,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://openrouter.ai/api/v1/chat/completions')
        expect(preview.headers).toMatchObject({
            Authorization: 'Bearer openrouter-key',
            'X-Title': 'RisuAI',
            'HTTP-Referer': 'https://risuai.xyz',
        })
        expect(preview.body).toMatchObject({
            model: 'free/model',
            route: 'fallback',
            transforms: ['middle-out'],
            provider: { order: ['Anthropic'], only: ['OpenAI'] },
        })
    })

    it('wires NanoGPT normal and subscription chat endpoints with provider headers', async () => {
        mocks.db.nanogptProvider = 'provider-a'

        const normal = await requestOpenAI(baseArg({ aiModel: 'nanogpt', previewBody: true }))
        expect(normal.type).toBe('success')
        let preview = JSON.parse(normal.result as string)
        expect(preview.url).toBe('https://nano-gpt.com/api/v1/chat/completions')
        expect(preview.body.model).toBe('nanogpt-model')
        expect(preview.headers.Authorization).toBe('Bearer nanogpt-key')
        expect(preview.headers['X-Provider']).toBe('provider-a')

        mocks.db.nanogptProvider = 'provider-sub'
        mocks.db.nanogptUseSubscriptionEndpoint = true
        const subscription = await requestOpenAI(baseArg({ aiModel: 'nanogpt', previewBody: true }))
        expect(subscription.type).toBe('success')
        preview = JSON.parse(subscription.result as string)
        expect(preview.url).toBe('https://nano-gpt.com/api/subscription/v1/chat/completions')
        expect(preview.headers['X-Provider']).toBe('provider-sub')
    })

    it('applies OpenAI Flex processing only for official OpenAI-compatible destinations', async () => {
        mocks.db.openAIFlexProcessing = true

        const official = await requestOpenAI(baseArg({ previewBody: true }))
        expect(official.type).toBe('success')
        expect(JSON.parse(official.result as string).body.service_tier).toBe('flex')

        const reverseOfficial = await requestOpenAI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'https://api.openai.com/v1/chat/completions',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                provider: LLMProvider.AsIs,
            },
        }))
        expect(reverseOfficial.type).toBe('success')
        expect(JSON.parse(reverseOfficial.result as string).body.service_tier).toBe('flex')

        const reverseOther = await requestOpenAI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'https://proxy.example/v1/chat/completions',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                provider: LLMProvider.AsIs,
            },
        }))
        expect(reverseOther.type).toBe('success')
        expect(JSON.parse(reverseOther.result as string).body.service_tier).toBeUndefined()
    })

    it('lets additional params override and delete Flex service_tier', async () => {
        mocks.db.openAIFlexProcessing = true
        mocks.db.additionalParams = [['service_tier', 'auto']]

        const overridden = await requestOpenAI(baseArg({ previewBody: true }))
        expect(overridden.type).toBe('success')
        expect(JSON.parse(overridden.result as string).body.service_tier).toBe('auto')

        mocks.db.additionalParams = [['service_tier', '{{none}}']]
        const deleted = await requestOpenAI(baseArg({ previewBody: true }))
        expect(deleted.type).toBe('success')
        expect(JSON.parse(deleted.result as string).body).not.toHaveProperty('service_tier')
    })

    it('returns the current failure contract for MultiGen with tools', async () => {
        const result = await requestOpenAI(baseArg({
            multiGen: true,
            tools: [{ name: 'lookup', description: 'Lookup', inputSchema: { type: 'object' } }],
        }))

        expect(result).toEqual({
            type: 'fail',
            result: 'MultiGen mode cannot be used with tool calls. Please disable one of them.',
        })
        expect(mocks.globalFetch).not.toHaveBeenCalled()
    })

    it('removes logit_bias for inlay-image models that are not GPT-compatible', async () => {
        mocks.supportsInlayImage.mockReturnValue(true)

        const result = await requestOpenAI(baseArg({
            aiModel: 'xcustom:::vision-model',
            biasString: [['soft bias', 3]],
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                endpoint: 'https://vision.example/v1/chat/completions',
                internalID: 'vision-model',
                provider: LLMProvider.AsIs,
            },
        }))

        expect(result.type).toBe('success')
        expect(JSON.parse(result.result as string).body.logit_bias).toBeUndefined()
    })

    it('sends non-streaming chat requests through globalFetch and extracts message text', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { choices: [{ message: { content: 'plain text' } }] },
        })

        const result = await requestOpenAI(baseArg())

        expect(result).toEqual({ type: 'success', result: 'plain text' })
        expect(mocks.db.cipherChat).toBe(false)
        expect(mocks.globalFetch).toHaveBeenCalledWith('https://api.openai.com/v1/chat/completions', expect.objectContaining({
            body: expect.objectContaining({ model: 'gpt-5', stream: false }),
            headers: expect.objectContaining({ Authorization: 'Bearer openai-key' }),
            interceptor: 'openai_basic',
        }))
    })

    it('extracts assistant content from a real-shaped non-streaming Chat Completions fixture', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: structuredClone(openAIChatCompletionFixture),
        })

        const result = await requestOpenAI(baseArg())

        expect(result).toEqual({ type: 'success', result: 'fixture ok' })
        expect(mocks.globalFetch).toHaveBeenCalledWith('https://api.openai.com/v1/chat/completions', expect.objectContaining({
            body: expect.objectContaining({ model: 'gpt-5', stream: false }),
            interceptor: 'openai_basic',
        }))
    })

    it('extracts readable HTTP error messages from non-streaming failures', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: false,
            data: { error: { message: 'bad request' } },
        })

        const result = await requestOpenAI(baseArg())

        expect(result).toEqual({ type: 'fail', result: 'HTTP bad request' })
    })

    it('extracts readable failure text from a real-shaped OpenAI Chat Completions error fixture', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: false,
            data: structuredClone(openAIChatErrorFixture),
        })

        const result = await requestOpenAI(baseArg())

        expect(result).toEqual({
            type: 'fail',
            result: 'HTTP The model `not-a-real-openai-model-for-fixtures` does not exist or you do not have access to it.',
        })
    })

    it('returns multiline results for non-streaming MultiGen choices', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { choices: [{ message: { content: 'one' } }, { message: { content: 'two' } }] },
        })

        const result = await requestOpenAI(baseArg({ multiGen: true }))

        expect(result).toEqual({ type: 'multiline', result: [['char', 'one'], ['char', 'two']] })
        expect(mocks.globalFetch.mock.calls[0][1].body.n).toBe(2)
    })

    it('wraps non-streaming reasoning fields in Thoughts output', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { choices: [{ message: { content: 'answer', reasoning_content: 'because' } }] },
        })

        const result = await requestOpenAI(baseArg())

        expect(result).toEqual({ type: 'success', result: '<Thoughts>\nbecause\n</Thoughts>\nanswer' })
    })

    it('wraps real-shaped DeepSeek reasoner reasoning_content in Thoughts output', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: structuredClone(deepSeekReasonerFixture),
        })

        const result = await requestOpenAI(baseArg({ aiModel: 'deepseek-reasoner' }))

        expect(result).toEqual({
            type: 'success',
            result: '<Thoughts>\nWe need to answer with only the word OK after reasoning briefly. The instruction is to reason briefly and then output "OK". No other text. So I\'ll just think and then say OK.\n</Thoughts>\nOK',
        })
    })

    it('wraps real-shaped OpenRouter reasoning fields in Thoughts output', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: structuredClone(openRouterReasoningFixture),
        })

        const result = await requestOpenAI(baseArg({ aiModel: 'openrouter' }))

        expect(result).toEqual({
            type: 'success',
            result: '<Thoughts>\nHmm, the user wants me to answer with only "OK" after some brief reasoning.\n</Thoughts>\nOK',
        })
    })

    it('continues non-streaming tool calls and appends remembered tool codes', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'tool result' }] as any)
        vi.mocked(encodeToolCall).mockResolvedValueOnce('<tool_call>encoded</tool_call>')
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: { choices: [{ message: { role: 'assistant', content: 'need tool', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{"q":"x"}' } }] } }] },
            })
            .mockResolvedValueOnce({
                ok: true,
                data: { choices: [{ message: { content: 'final answer' } }] },
            })

        const result = await requestOpenAI(baseArg({
            rememberToolUsage: true,
            tools: [{ name: 'lookup', description: 'Lookup', inputSchema: { type: 'object', properties: { q: { type: 'string' } } } }],
        }))

        expect(result).toEqual({ type: 'success', result: 'need tool\n\n<tool_call>encoded</tool_call>\n\nfinal answer' })
        expect(callTool).toHaveBeenCalledWith('lookup', { q: 'x' })
        expect(mocks.globalFetch).toHaveBeenCalledTimes(2)
        const followUpMessages = mocks.globalFetch.mock.calls[1][1].body.messages
        expect(followUpMessages.at(-2)).toMatchObject({ role: 'assistant', tool_calls: expect.any(Array) })
        expect(followUpMessages.at(-1)).toEqual({ role: 'tool', content: 'tool result', tool_call_id: 'call_1' })
    })

    it('continues tool calls from a real-shaped Chat Completions tool-call fixture', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'fixture tool result' }] as any)
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: structuredClone(openAIChatToolCallFixture),
            })
            .mockResolvedValueOnce({
                ok: true,
                data: structuredClone(openAIChatCompletionFixture),
            })

        const result = await requestOpenAI(baseArg({
            tools: [{ name: 'fixture_lookup', description: 'Fixture lookup', inputSchema: { type: 'object', properties: { query: { type: 'string' } } } }],
        }))

        expect(result).toEqual({ type: 'success', result: '\n\n\n\nfixture ok' })
        expect(callTool).toHaveBeenCalledWith('fixture_lookup', { query: 'abc' })
        expect(mocks.globalFetch).toHaveBeenCalledTimes(2)
        const followUpMessages = mocks.globalFetch.mock.calls[1][1].body.messages
        expect(followUpMessages.at(-2)).toEqual(openAIChatToolCallFixture.choices[0].message)
        expect(followUpMessages.at(-1)).toEqual({
            role: 'tool',
            content: 'fixture tool result',
            tool_call_id: 'call_fixture_lookup',
        })
    })

    it('falls back to accumulated non-streaming tool prefix when follow-up fails', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'tool result' }] as any)
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: { choices: [{ message: { role: 'assistant', content: 'partial', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] } }] },
            })
            .mockResolvedValueOnce({
                ok: false,
                data: { error: { message: 'follow-up failed' } },
            })

        const result = await requestOpenAI(baseArg({
            tools: [{ name: 'lookup', description: 'Lookup', inputSchema: { type: 'object' } }],
        }))

        expect(result).toEqual({ type: 'success', result: 'partial\n\n' })
        expect(alertError).toHaveBeenCalledWith('Failed to fetch model response after tool execution')
    })

    it('adds fallback tool messages for unknown tools and malformed arguments', async () => {
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: {
                    choices: [{
                        message: {
                            role: 'assistant',
                            content: 'need tools',
                            tool_calls: [
                                { id: 'call_unknown', type: 'function', function: { name: 'missing', arguments: '{"q":"x"}' } },
                                { id: 'call_bad_json', type: 'function', function: { name: 'lookup', arguments: '{bad' } },
                            ],
                        },
                    }],
                },
            })
            .mockResolvedValueOnce({
                ok: true,
                data: { choices: [{ message: { content: 'final answer' } }] },
            })

        const result = await requestOpenAI(baseArg({
            tools: [{ name: 'lookup', description: 'Lookup', inputSchema: { type: 'object' } }],
        }))

        expect(result).toEqual({ type: 'success', result: 'need tools\n\n\n\nfinal answer' })
        expect(callTool).not.toHaveBeenCalled()
        const followUpMessages = mocks.globalFetch.mock.calls[1][1].body.messages
        expect(followUpMessages).toHaveLength(5)
        expect(followUpMessages.slice(-3)).toEqual([
            {
                role: 'assistant',
                content: 'need tools',
                tool_calls: [
                    { id: 'call_unknown', type: 'function', function: { name: 'missing', arguments: '{"q":"x"}' } },
                    { id: 'call_bad_json', type: 'function', function: { name: 'lookup', arguments: '{bad' } },
                ],
            },
            {
                role: 'tool',
                content: 'No tool found with name: missing',
                tool_call_id: 'call_unknown',
            },
            {
                role: 'tool',
                content: expect.stringContaining('Tool call failed with error: SyntaxError'),
                tool_call_id: 'call_bad_json',
            },
        ])
    })

    it('returns streaming preview bodies with stream enabled', async () => {
        const result = await requestOpenAI(baseArg({ previewBody: true, useStreaming: true }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.body.stream).toBe(true)
        expect(preview.url).toBe('https://api.openai.com/v1/chat/completions')
    })

    it('returns readable failures for streaming HTTP and content-type errors', async () => {
        const httpResponse = streamResponse(['bad request'], 500)
        mocks.fetchNative.mockResolvedValueOnce(httpResponse)

        const httpError = await requestOpenAI(baseArg({ useStreaming: true }))
        expect(httpError).toEqual({ type: 'fail', result: 'stream error body' })
        expect(textifyReadableStream).toHaveBeenNthCalledWith(1, httpResponse.body)

        const contentTypeResponse = streamResponse(['{"error":"not sse"}'], 200, 'application/json')
        mocks.fetchNative.mockResolvedValueOnce(contentTypeResponse)

        const contentTypeError = await requestOpenAI(baseArg({ useStreaming: true }))
        expect(contentTypeError).toEqual({ type: 'fail', result: 'stream error body' })
        expect(textifyReadableStream).toHaveBeenNthCalledWith(2, contentTypeResponse.body)
    })

    it('sends streaming chat requests through fetchNative and collects SSE text', async () => {
        mocks.fetchNative.mockResolvedValueOnce(streamResponse([
            'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n',
            'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n',
            'data: [DONE]\n\n',
        ]))

        const result = await requestOpenAI(baseArg({ useStreaming: true }))

        expect(result.type).toBe('streaming')
        expect(fetchNative).toHaveBeenCalledWith('https://api.openai.com/v1/chat/completions', expect.objectContaining({
            method: 'POST',
            headers: expect.objectContaining({ Authorization: 'Bearer openai-key' }),
            interceptor: 'openai_streaming',
        }))
        const sentBody = JSON.parse(mocks.fetchNative.mock.calls[0][1].body)
        expect(sentBody.stream).toBe(true)
        const chunks = await collectStream(result.result as ReadableStream<Record<string, string>>)
        expect(chunks.at(-1)).toEqual({ '0': 'Hello' })
    })

    it('accumulates streamed content from real-shaped Chat Completions SSE chunk fixtures', async () => {
        mocks.fetchNative.mockResolvedValueOnce(streamResponse([...openAIChatStreamingEventsFixture]))

        const result = await requestOpenAI(baseArg({ useStreaming: true }))

        expect(result.type).toBe('streaming')
        expect(fetchNative).toHaveBeenCalledWith('https://api.openai.com/v1/chat/completions', expect.objectContaining({
            method: 'POST',
            interceptor: 'openai_streaming',
        }))
        const chunks = await collectStream(result.result as ReadableStream<Record<string, string>>)
        expect(chunks[0]).toEqual({ '0': '' })
        expect(chunks.at(-1)).toEqual({ '0': 'stream ok' })
    })

    it('currently emits an empty chunk for malformed streaming SSE payloads before preserving valid chunks', async () => {
        mocks.fetchNative.mockResolvedValueOnce(streamResponse([
            'data: not-json\n\n',
            'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n',
            'data: [DONE]\n\n',
        ]))

        const result = await requestOpenAI(baseArg({ useStreaming: true }))

        expect(result.type).toBe('streaming')
        const chunks = await collectStream(result.result as ReadableStream<Record<string, string>>)
        expect(chunks).toEqual([{ '0': '' }, { '0': 'ok' }, { '0': 'ok' }])
        expect(chunks.at(-1)).toEqual({ '0': 'ok' })
    })

    it('passes local-network route metadata to non-streaming and streaming requests', async () => {
        mocks.db.localNetworkMode = true
        mocks.db.localNetworkTimeoutSec = 12
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { choices: [{ message: { content: 'local' } }] },
        })

        const localURL = 'http://192.168.1.25:11434/v1/chat/completions'
        const nonStreaming = await requestOpenAI(baseArg({ customURL: localURL }))

        expect(nonStreaming).toEqual({ type: 'success', result: 'local' })
        expect(mocks.globalFetch).toHaveBeenCalledWith(localURL, expect.objectContaining({
            networkRoute: 'local_network',
            requestTimeoutMs: undefined,
        }))

        mocks.fetchNative.mockResolvedValueOnce(streamResponse(['data: [DONE]\n\n']))

        const streaming = await requestOpenAI(baseArg({ customURL: localURL, useStreaming: true }))

        expect(streaming.type).toBe('streaming')
        expect(mocks.fetchNative).toHaveBeenCalledWith(localURL, expect.objectContaining({
            networkRoute: 'local_network',
            requestTimeoutMs: 12000,
        }))
        await collectStream(streaming.result as ReadableStream<Record<string, string>>)
    })

    it('omits local-network metadata for local URLs without local mode and remote URLs with local mode', async () => {
        const localURL = 'http://192.168.1.25:11434/v1/chat/completions'
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { choices: [{ message: { content: 'local without mode' } }] },
        })

        const localWithoutMode = await requestOpenAI(baseArg({ customURL: localURL }))

        expect(localWithoutMode).toEqual({ type: 'success', result: 'local without mode' })
        expect(mocks.globalFetch).toHaveBeenCalledWith(localURL, expect.objectContaining({
            networkRoute: undefined,
            requestTimeoutMs: undefined,
        }))

        mocks.fetchNative.mockResolvedValueOnce(streamResponse(['data: [DONE]\n\n']))

        const localStreamingWithoutMode = await requestOpenAI(baseArg({ customURL: localURL, useStreaming: true }))

        expect(localStreamingWithoutMode.type).toBe('streaming')
        expect(mocks.fetchNative).toHaveBeenCalledWith(localURL, expect.objectContaining({
            networkRoute: undefined,
            requestTimeoutMs: undefined,
        }))
        await collectStream(localStreamingWithoutMode.result as ReadableStream<Record<string, string>>)

        mocks.db.localNetworkMode = true
        const remoteURL = 'https://remote.example/v1/chat/completions'
        mocks.globalFetch.mockClear()
        mocks.fetchNative.mockClear()
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { choices: [{ message: { content: 'remote with mode' } }] },
        })

        const remoteWithMode = await requestOpenAI(baseArg({ customURL: remoteURL }))

        expect(remoteWithMode).toEqual({ type: 'success', result: 'remote with mode' })
        expect(mocks.globalFetch).toHaveBeenCalledWith(remoteURL, expect.objectContaining({
            networkRoute: undefined,
            requestTimeoutMs: undefined,
        }))

        mocks.fetchNative.mockResolvedValueOnce(streamResponse(['data: [DONE]\n\n']))

        const remoteStreamingWithMode = await requestOpenAI(baseArg({ customURL: remoteURL, useStreaming: true }))

        expect(remoteStreamingWithMode.type).toBe('streaming')
        expect(mocks.fetchNative).toHaveBeenCalledWith(remoteURL, expect.objectContaining({
            networkRoute: undefined,
            requestTimeoutMs: undefined,
        }))
        await collectStream(remoteStreamingWithMode.result as ReadableStream<Record<string, string>>)
    })

    it('continues streaming tool calls with fetchNative and emits prefix plus final text', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'tool result' }] as any)
        mocks.fetchNative
            .mockResolvedValueOnce(streamResponse([
                'data: {"choices":[{"index":0,"delta":{"content":"need tool"}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\\\"q\\\":\\\"x\\\"}"}}]}}]}\n\n',
                'data: [DONE]\n\n',
            ]))
            .mockResolvedValueOnce(streamResponse([
                'data: {"choices":[{"index":0,"delta":{"content":"final"}}]}\n\n',
                'data: [DONE]\n\n',
            ]))

        const result = await requestOpenAI(baseArg({
            useStreaming: true,
            tools: [{ name: 'lookup', description: 'Lookup', inputSchema: { type: 'object', properties: { q: { type: 'string' } } } }],
        }))

        expect(result.type).toBe('streaming')
        const chunks = await collectStream(result.result as ReadableStream<Record<string, string>>)
        expect(callTool).toHaveBeenCalledWith('lookup', { q: 'x' })
        expect(mocks.fetchNative).toHaveBeenCalledTimes(2)
        expect(mocks.fetchNative.mock.calls[1][1].interceptor).toBe('openai_tool')
        expect(chunks).toContainEqual({ '0': 'need tool\n\n' })
        expect(chunks.at(-1)).toEqual({ '0': 'need tool\n\n\n\nfinal' })
    })

    it('adds streaming fallback tool messages for unknown tools before follow-up output', async () => {
        mocks.fetchNative
            .mockResolvedValueOnce(streamResponse([
                'data: {"choices":[{"index":0,"delta":{"content":"need missing"}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_missing","function":{"name":"missing","arguments":"{\\"q\\":\\"x\\"}"}}]}}]}\n\n',
                'data: [DONE]\n\n',
            ]))
            .mockResolvedValueOnce(streamResponse([
                'data: {"choices":[{"index":0,"delta":{"content":"final"}}]}\n\n',
                'data: [DONE]\n\n',
            ]))

        const result = await requestOpenAI(baseArg({
            useStreaming: true,
            tools: [{ name: 'lookup', description: 'Lookup', inputSchema: { type: 'object' } }],
        }))

        expect(result.type).toBe('streaming')
        const chunks = await collectStream(result.result as ReadableStream<Record<string, string>>)
        expect(callTool).not.toHaveBeenCalled()
        expect(mocks.fetchNative).toHaveBeenCalledTimes(2)
        expect(mocks.fetchNative.mock.calls[1][1].interceptor).toBe('openai_tool')
        const followUpMessages = JSON.parse(mocks.fetchNative.mock.calls[1][1].body).messages
        expect(followUpMessages).toHaveLength(4)
        expect(followUpMessages.slice(-2)).toEqual([
            {
                role: 'assistant',
                content: 'need missing',
                tool_calls: [{ id: 'call_missing', type: 'function', function: { name: 'missing', arguments: '{"q":"x"}' } }],
            },
            {
                role: 'tool',
                content: 'No tool found with name: missing',
                tool_call_id: 'call_missing',
            },
        ])
        expect(chunks).toContainEqual({ '0': 'need missing\n\n' })
        expect(chunks.at(-1)).toEqual({ '0': 'need missing\n\n\n\nfinal' })
    })

    it('currently allows browser streaming to 127.0.0.1 while blocking localhost', async () => {
        mocks.platform.isNodeServer = false
        mocks.platform.isTauri = false
        mocks.fetchNative.mockResolvedValueOnce(streamResponse(['data: [DONE]\n\n']))

        const loopbackIP = await requestOpenAI(baseArg({ customURL: 'http://127.0.0.1:11434/v1/chat/completions', useStreaming: true }))
        expect(loopbackIP.type).toBe('streaming')
        await collectStream(loopbackIP.result as ReadableStream<Record<string, string>>)

        const localhost = await requestOpenAI(baseArg({ customURL: 'http://localhost:11434/v1/chat/completions', useStreaming: true }))
        expect(localhost).toEqual({
            type: 'fail',
            result: 'You are trying local request on streaming. this is not allowed dude to browser/os security policy. turn off streaming.',
        })
    })
})
