import { beforeEach, describe, expect, it, vi } from 'vitest'

import { LLMFlags, LLMFormat, LLMProvider, LLMTokenizer } from 'src/ts/model/types'
import { fetchNative } from 'src/ts/globalApi.svelte'
import { alertError } from 'src/ts/alert'
import { callTool } from '../../mcp/mcp'
import { __testResponsesAPI, requestOpenAIResponseAPI } from './requests'
import { openAIResponsesErrorFixture, openAIResponsesStreamingEventsFixture, openAIResponsesSuccessFixture, openAIResponsesToolCallFixture } from './fixtures/responses'

const mocks = vi.hoisted(() => ({
    db: {
        OaiCompAPIKeys: {},
        additionalParams: [],
        autofillRequestUrl: false,
        customModels: [],
        gptVisionQuality: 'high',
        jsonSchema: '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}',
        jsonSchemaEnabled: false,
        localNetworkMode: false,
        modelTools: [] as string[],
        newOAIHandle: true,
        nanogptKey: 'nanogpt-key',
        nanogptProvider: '',
        nanogptRequestModel: 'nanogpt-model',
        nanogptUseSubscriptionEndpoint: false,
        openAIKey: 'openai-key',
        openAIFlexProcessing: false,
        proxyKey: 'proxy-key',
        requestRetrys: 0,
        reasoningEffort: 2,
        seperateParametersEnabled: false,
        simplifiedToolUse: false,
        strictJsonSchema: true,
        temperature: 70,
        top_p: 0.9,
        verbosity: 0,
    },
    fetchNative: vi.fn(),
    globalFetch: vi.fn(),
    isLocalNetworkUrl: vi.fn((url: string) => {
        try{
            const host = new URL(url).hostname.trim().toLowerCase().replace(/^\[|\]$/g, '').split('%')[0]
            const parts = host.split('.').map((part) => Number(part))
            const privateIPv4 = parts.length === 4 && parts.every((part) => Number.isInteger(part) && part >= 0 && part <= 255) && (
                parts[0] === 10 ||
                (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) ||
                (parts[0] === 192 && parts[1] === 168) ||
                (parts[0] === 169 && parts[1] === 254)
            )
            const firstHextet = Number.parseInt(host.split(':')[0], 16)
            const localIPv6 = host === '::1' || (Number.isFinite(firstHextet) && (firstHextet >= 0xfc00 && firstHextet <= 0xfdff || firstHextet >= 0xfe80 && firstHextet <= 0xfebf))
            const localDNS = host.endsWith('.local') || (/^[a-z0-9_-]+$/i.test(host) && !host.includes('.'))
            return host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0' || privateIPv4 || localIPv6 || localDNS
        }
        catch{
            return false
        }
    }),
    platform: {
        isNodeServer: true,
        isTauri: false,
    },
}))

vi.mock('src/ts/storage/database.svelte', () => ({
    getDatabase: () => mocks.db,
}))

vi.mock('src/ts/globalApi.svelte', () => ({
    addFetchLog: vi.fn(),
    fetchNative: mocks.fetchNative,
    globalFetch: mocks.globalFetch,
    textifyReadableStream: vi.fn(),
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

vi.mock('src/ts/network/localNetwork', () => ({
    isLocalNetworkUrl: mocks.isLocalNetworkUrl,
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
    applyChatTemplate: vi.fn(),
}))

vi.mock('src/ts/process/templates/chatTemplate', () => ({
    applyChatTemplate: vi.fn(),
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
        OpenAIResponseAPI: 18,
        NanoGPTResponses: 21,
    },
    LLMProvider: {
        OpenAI: 0,
        AsIs: 4,
    },
    getFreeOpenRouterModels: vi.fn(),
}))

vi.mock('src/ts/tokenizer', () => ({
    strongBan: vi.fn(),
    tokenizeNum: vi.fn(),
}))

vi.mock('src/ts/model/openrouter', () => ({
    getFreeOpenRouterModels: vi.fn(),
}))

vi.mock('src/ts/util', () => ({
    simplifySchema: (schema: unknown) => schema,
}))

vi.mock('../../files/inlays', () => ({
    supportsInlayImage: () => false,
}))

vi.mock('../../mcp/mcp', () => ({
    callTool: vi.fn(),
    decodeToolCall: vi.fn(),
    encodeToolCall: vi.fn(),
}))

const baseArg = (overrides: Record<string, any> = {}) => ({
    aiModel: 'gpt-5-response-api',
    bias: {},
    biasString: [],
    formated: [
        { role: 'system', content: 'Follow policy.' },
        { role: 'user', content: 'Describe this.', multimodals: [{ type: 'image', base64: 'data:image/png;base64,abc' }] },
        { role: 'assistant', content: 'Previous assistant prefill' },
        { role: 'user', content: 'Also read file.', multimodals: [{ type: 'audio', base64: 'data:application/pdf;base64,def' }] },
    ],
    maxTokens: 321,
    mode: 'model',
    modelInfo: {
        flags: [LLMFlags.DeveloperRole],
        format: LLMFormat.OpenAIResponseAPI,
        id: 'gpt-5-response-api',
        internalID: 'gpt-5',
        name: 'GPT-5 Responses',
        parameters: ['temperature', 'top_p', 'reasoning_effort', 'verbosity'],
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

describe('OpenAI Responses API helpers', () => {
    beforeEach(() => {
        mocks.fetchNative.mockReset()
        mocks.globalFetch.mockReset()
        mocks.db.OaiCompAPIKeys = {}
        mocks.db.additionalParams = []
        mocks.db.jsonSchemaEnabled = false
        mocks.db.modelTools = []
        mocks.db.customModels = []
        mocks.db.nanogptProvider = ''
        mocks.db.nanogptRequestModel = 'nanogpt-model'
        mocks.db.nanogptUseSubscriptionEndpoint = false
        mocks.db.simplifiedToolUse = false
        mocks.db.autofillRequestUrl = false
        mocks.db.openAIFlexProcessing = false
        mocks.db.requestRetrys = 0
        mocks.isLocalNetworkUrl.mockClear()
        mocks.platform.isNodeServer = true
        mocks.platform.isTauri = false
        vi.mocked(alertError).mockReset()
    })

    it('builds a Responses request body for text, developer role, multimodal input, tools, and model parameters', async () => {
        mocks.db.modelTools = ['search']
        const sourceMessages = baseArg().formated

        const body = await __testResponsesAPI.buildResponsesBody(baseArg({
            formated: sourceMessages,
            schema: '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}',
            tools: [{ name: 'lookup', description: 'Lookup data', inputSchema: { type: 'object', properties: { query: { type: 'string' } } } }],
        }))

        expect(body).toMatchObject({
            model: 'gpt-5',
            max_output_tokens: 321,
            store: false,
            temperature: 0.7,
            top_p: 0.9,
            reasoning: { effort: 'high', summary: 'auto' },
            text: {
                verbosity: 'low',
                format: {
                    type: 'json_schema',
                    name: 'format',
                    strict: true,
                },
            },
        })
        expect(body.input[0]).toMatchObject({ role: 'developer', content: [{ type: 'input_text', text: 'Follow policy.' }] })
        expect(body.input[1].content).toEqual([
            { type: 'input_text', text: 'Describe this.' },
            { type: 'input_image', detail: 'high', image_url: 'data:image/png;base64,abc' },
        ])
        expect(body.input[2]).toMatchObject({
            role: 'assistant',
            type: 'message',
            content: [{ type: 'output_text', text: 'Previous assistant prefill', annotations: [] }],
        })
        expect(body.input[3].content).toEqual([
            { type: 'input_text', text: 'Also read file.' },
            { type: 'input_file', file_data: 'data:application/pdf;base64,def' },
        ])
        expect(body.tools).toEqual([
            { type: 'function', name: 'lookup', description: 'Lookup data', parameters: { type: 'object', properties: { query: { type: 'string' } } } },
            { type: 'web_search_preview' },
        ])
        expect(sourceMessages[1]).toMatchObject({ role: 'user', content: 'Describe this.', multimodals: [{ type: 'image', base64: 'data:image/png;base64,abc' }] })
    })

    it('requests reasoning summaries for Responses reasoning models', async () => {
        const body = await __testResponsesAPI.buildResponsesBody(baseArg())

        expect(body.reasoning).toEqual({ effort: 'high', summary: 'auto' })
    })

    it('does not request reasoning summaries for Responses non-reasoning models', async () => {
        const body = await __testResponsesAPI.buildResponsesBody(baseArg({
            modelInfo: {
                ...baseArg().modelInfo,
                parameters: ['temperature', 'top_p', 'verbosity'],
            },
        }))

        expect(body.reasoning?.summary).toBeUndefined()
    })

    it('lets reverse proxy additional params override Responses reasoning summaries', async () => {
        mocks.db.additionalParams = [
            ['reasoning.summary', 'detailed'],
        ]

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'https://proxy.example/v1/responses',
            previewBody: true,
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.body.reasoning).toEqual({ effort: 'high', summary: 'detailed' })
    })

    it('applies global additional params to official OpenAI Responses requests', async () => {
        mocks.db.additionalParams = [
            ['metadata.source', 'global'],
            ['include', 'json::["message.output_text"]'],
        ]

        const result = await requestOpenAIResponseAPI(baseArg({ previewBody: true }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.body.metadata.source).toBe('global')
        expect(preview.body.include).toEqual(['message.output_text'])
    })

    it('uses OpenAI defaults and an externally clean non-streaming body', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { output_text: 'ok' },
        })

        const result = await requestOpenAIResponseAPI(baseArg())

        expect(result).toEqual({ type: 'success', result: 'ok' })
        expect(mocks.globalFetch).toHaveBeenCalledWith('https://api.openai.com/v1/responses', expect.objectContaining({
            body: expect.objectContaining({
                model: 'gpt-5',
                stream: false,
            }),
            headers: expect.objectContaining({ Authorization: 'Bearer openai-key' }),
        }))
        expect(mocks.globalFetch.mock.calls[0][1].body).not.toHaveProperty('__lastOutput')
    })

    it('extracts assistant text from a real-shaped non-streaming Responses fixture', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: structuredClone(openAIResponsesSuccessFixture),
        })

        const result = await requestOpenAIResponseAPI(baseArg())

        expect(result).toEqual({ type: 'success', result: 'response fixture ok' })
        expect(mocks.globalFetch).toHaveBeenCalledTimes(1)
        expect(mocks.globalFetch.mock.calls[0][1].body.input[1].content).toEqual([
            { type: 'input_text', text: 'Describe this.' },
            { type: 'input_image', detail: 'high', image_url: 'data:image/png;base64,abc' },
        ])
        expect(openAIResponsesSuccessFixture).toMatchObject({
            object: 'response',
            status: 'completed',
            billing: { payer: 'developer' },
            completed_at: 1700000001,
            reasoning: { context: null, effort: null, summary: null },
            service_tier: 'default',
            usage: { total_tokens: 18 },
            text: { format: { type: 'text' }, verbosity: 'medium' },
            metadata: {},
        })
    })

    it('leaves system messages as system without the developer-role flag', async () => {
        const body = await __testResponsesAPI.buildResponsesBody(baseArg({
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
            },
        }))

        expect(body.input[0]).toMatchObject({ role: 'system', content: [{ type: 'input_text', text: 'Follow policy.' }] })
    })

    it('applies custom model Responses endpoint, key, and additional params', async () => {
        mocks.db.OaiCompAPIKeys = { customKey: 'custom-key' }
        mocks.db.additionalParams = [
            ['metadata.global', 'yes'],
        ]
        mocks.db.customModels = [{
            id: 'xcustom:::responses',
            params: 'header::X-Custom=yes\nmetadata.tier=gold\nextra=json::{"enabled":true}',
        }]

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'xcustom:::responses',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                endpoint: 'https://custom.example/v1/responses',
                keyIdentifier: 'customKey',
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://custom.example/v1/responses')
        expect(preview.headers.Authorization).toBe('Bearer custom-key')
        expect(preview.headers['X-Custom']).toBe('yes')
        expect(preview.body.metadata.global).toBe('yes')
        expect(preview.body.metadata.tier).toBe('gold')
        expect(preview.body.extra).toEqual({ enabled: true })
    })

    it('skips structured output for models flagged noStructuredOutput', async () => {
        const body = await __testResponsesAPI.buildResponsesBody(baseArg({
            schema: '{"type":"object"}',
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [LLMFlags.noStructuredOutput],
            },
        }))

        expect(body.text?.format).toBeUndefined()
    })

    it('keeps structured output for models without noStructuredOutput', async () => {
        const body = await __testResponsesAPI.buildResponsesBody(baseArg({
            schema: '{"type":"object"}',
            modelInfo: {
                ...baseArg().modelInfo,
                flags: [],
            },
        }))

        expect(body.text?.format).toMatchObject({ type: 'json_schema', name: 'format' })
    })

    it('does not duplicate top-level output_text when message output blocks are also present', () => {
        const text = __testResponsesAPI.extractResponsesText({
            output_text: 'final text',
            output: [{ type: 'message', content: [{ type: 'output_text', text: 'final text' }] }],
        }, baseArg())

        expect(text).toBe('final text')
    })

    it('preserves reasoning summaries and refusal-only responses', () => {
        expect(__testResponsesAPI.extractResponsesText({
            output: [
                { type: 'reasoning', summary: [{ text: 'reasoned' }] },
                { type: 'message', content: [{ type: 'output_text', text: 'answer' }] },
            ],
        }, baseArg())).toBe('<Thoughts>\n\nreasoned\n\n</Thoughts>\nanswer')

        expect(__testResponsesAPI.extractResponsesText({
            output: [{ type: 'message', content: [{ type: 'refusal', refusal: 'Cannot comply.' }] }],
        }, baseArg())).toBe('Cannot comply.')
    })

    it('extracts OpenRouter-style reasoning content reasoning_text with final output_text', () => {
        const text = __testResponsesAPI.extractResponsesText({
            output: [
                {
                    id: 'rs_tmp_15f15eqwfj4',
                    type: 'reasoning',
                    status: 'completed',
                    content: [{ type: 'reasoning_text', text: 'Hmm, the user just greeted me...' }],
                    summary: [],
                    format: 'unknown',
                },
                { type: 'message', content: [{ type: 'output_text', text: 'Hello there!' }] },
            ],
        }, baseArg())

        expect(text).toBe('<Thoughts>\n\nHmm, the user just greeted me...\n\n</Thoughts>\nHello there!')
    })

    it('does not add an empty thoughts block for OpenAI-style empty reasoning summaries', () => {
        const text = __testResponsesAPI.extractResponsesText({
            output: [
                { id: 'rs_0d1786ac1d609512016a07343fdca8819ca2d651a07a09a86d', type: 'reasoning', summary: [] },
                { type: 'message', content: [{ type: 'output_text', text: 'Only final answer.' }] },
            ],
        }, baseArg())

        expect(text).toBe('Only final answer.')
    })

    it('treats incomplete non-streaming Responses results as failures even when partial text exists', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: {
                status: 'incomplete',
                incomplete_details: { reason: 'max_output_tokens' },
                output_text: 'partial',
            },
        })

        const result = await requestOpenAIResponseAPI(baseArg())

        expect(result).toEqual({ type: 'fail', result: 'Incomplete response: max_output_tokens\npartial' })
    })

    it('treats failed non-streaming Responses results as useful failures', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: {
                status: 'failed',
                error: { message: 'bad request' },
            },
        })

        const result = await requestOpenAIResponseAPI(baseArg())

        expect(result).toEqual({ type: 'fail', result: 'bad request' })
    })

    it('extracts readable error messages from non-streaming HTTP errors', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: false,
            data: { error: { message: 'http bad request' } },
        })

        const result = await requestOpenAIResponseAPI(baseArg())

        expect(result).toEqual({ type: 'fail', result: 'HTTP http bad request' })
    })

    it('extracts readable invalid-model errors from a real-shaped Responses error fixture', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: false,
            data: structuredClone(openAIResponsesErrorFixture),
        })

        const result = await requestOpenAIResponseAPI(baseArg())

        expect(result).toEqual({
            type: 'fail',
            result: "HTTP The requested model 'not-a-real-openai-model-for-fixtures' does not exist.",
        })
        expect(mocks.globalFetch).toHaveBeenCalledWith('https://api.openai.com/v1/responses', expect.objectContaining({
            body: expect.objectContaining({ model: 'gpt-5', stream: false }),
        }))
    })

    it('extracts readable string-shaped Responses errors', async () => {
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: false,
                data: { error: 'http string bad request' },
            })
            .mockResolvedValueOnce({
                ok: true,
                data: { status: 'failed', error: 'failed string bad request' },
            })
            .mockResolvedValueOnce({
                ok: false,
                data: 'raw string bad request',
            })

        const httpObjectString = await requestOpenAIResponseAPI(baseArg())
        const failedString = await requestOpenAIResponseAPI(baseArg())
        const httpRawString = await requestOpenAIResponseAPI(baseArg())

        expect(httpObjectString).toEqual({ type: 'fail', result: 'HTTP http string bad request' })
        expect(failedString).toEqual({ type: 'fail', result: 'failed string bad request' })
        expect(httpRawString).toEqual({ type: 'fail', result: 'HTTP raw string bad request' })
    })

    it('falls back to JSON for unknown Responses error shapes', async () => {
        mocks.globalFetch.mockResolvedValueOnce({
            ok: true,
            data: { status: 'failed', code: 123 },
        })

        const result = await requestOpenAIResponseAPI(baseArg())

        expect(result).toEqual({ type: 'fail', result: '{"status":"failed","code":123}' })
    })

    it('handles an omitted aiModel in the Responses path without crashing', async () => {
        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: undefined,
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                internalID: undefined,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://api.openai.com/v1/responses')
        expect(preview.body.model).toBe('gpt-4.1')
    })

    it('wires NanoGPT Responses endpoint, model, auth, and provider header', async () => {
        mocks.db.nanogptProvider = 'provider-a'

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'nanogpt',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                internalID: 'nanogpt',
                format: LLMFormat.NanoGPTResponses,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://nano-gpt.com/api/v1/responses')
        expect(preview.body.model).toBe('nanogpt-model')
        expect(preview.headers.Authorization).toBe('Bearer nanogpt-key')
        expect(preview.headers['X-Provider']).toBe('provider-a')
    })

    it('sends NanoGPT provider header on the subscription Responses endpoint', async () => {
        mocks.db.nanogptProvider = 'provider-sub'
        mocks.db.nanogptUseSubscriptionEndpoint = true

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'nanogpt',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                internalID: 'nanogpt',
                format: LLMFormat.NanoGPTResponses,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://nano-gpt.com/api/subscription/v1/responses')
        expect(preview.headers['X-Provider']).toBe('provider-sub')
    })

    it('applies OpenAI Flex processing to official Responses requests', async () => {
        mocks.db.openAIFlexProcessing = true

        const result = await requestOpenAIResponseAPI(baseArg({ previewBody: true }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.body.service_tier).toBe('flex')
    })

    it('applies OpenAI Flex processing to reverse proxy requests pointed at official OpenAI', async () => {
        mocks.db.openAIFlexProcessing = true

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'https://api.openai.com/v1/responses',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                provider: LLMProvider.AsIs,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.body.service_tier).toBe('flex')
    })

    it('does not apply OpenAI Flex processing to reverse proxy requests pointed away from OpenAI', async () => {
        mocks.db.openAIFlexProcessing = true

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'https://proxy.example/v1/responses',
            previewBody: true,
            modelInfo: {
                ...baseArg().modelInfo,
                provider: LLMProvider.AsIs,
            },
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.body.service_tier).toBeUndefined()
    })

    it('lets additional params override or delete OpenAI Flex service_tier', async () => {
        mocks.db.openAIFlexProcessing = true
        mocks.db.additionalParams = [['service_tier', 'auto']]

        const overridden = await requestOpenAIResponseAPI(baseArg({ previewBody: true }))
        expect(overridden.type).toBe('success')
        expect(JSON.parse(overridden.result as string).body.service_tier).toBe('auto')

        mocks.db.additionalParams = [['service_tier', '{{none}}']]

        const deleted = await requestOpenAIResponseAPI(baseArg({ previewBody: true }))
        expect(deleted.type).toBe('success')
        expect(JSON.parse(deleted.result as string).body).not.toHaveProperty('service_tier')
    })

    it('lets additional params delete nested Responses defaults', async () => {
        mocks.db.additionalParams = [['reasoning.summary', '{{none}}']]

        const result = await requestOpenAIResponseAPI(baseArg({ previewBody: true }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.body.reasoning).toEqual({ effort: 'high' })
    })

    it('applies reverse proxy Responses endpoint autofill and additional params', async () => {
        mocks.db.autofillRequestUrl = true
        mocks.db.additionalParams = [
            ['header::X-Test', 'ok'],
            ['metadata.source', 'risu'],
        ]

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'https://proxy.example/api',
            previewBody: true,
            key: undefined,
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://proxy.example/api/v1/responses')
        expect(preview.headers.Authorization).toBe('Bearer proxy-key')
        expect(preview.headers['X-Test']).toBe('ok')
        expect(preview.body.metadata.source).toBe('risu')
    })

    it('preserves risu proxy identify headers for Responses reverse proxy requests', async () => {
        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'reverse_proxy',
            customURL: 'risu::https://proxy.example/v1/responses',
            previewBody: true,
            key: undefined,
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('https://proxy.example/v1/responses')
        expect(preview.headers.Authorization).toBe('Bearer proxy-key')
        expect(preview.headers['X-Proxy-Risu']).toBe('RisuAI')
    })

    it.each([
        ['https://proxy.example/v1', 'https://proxy.example/v1/responses'],
        ['https://proxy.example/v1/', 'https://proxy.example/v1/responses'],
        ['https://proxy.example/api', 'https://proxy.example/api/v1/responses'],
        ['https://proxy.example/v1/responses', 'https://proxy.example/v1/responses'],
        ['https://azure.example/openai/deployments/model/responses?api-version=2025-04-01-preview', 'https://azure.example/openai/deployments/model/responses?api-version=2025-04-01-preview'],
    ])('autofills reverse proxy Responses URL variant %s', async (customURL, expectedURL) => {
        mocks.db.autofillRequestUrl = true

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'reverse_proxy',
            customURL,
            previewBody: true,
            key: undefined,
        }))

        expect(result.type).toBe('success')
        expect(JSON.parse(result.result as string).url).toBe(expectedURL)
    })

    it.each([
        ['not a url?api-version=2025-04-01-preview', 'not a url/v1/responses?api-version=2025-04-01-preview'],
        ['not a url/v1', 'not a url/v1/responses'],
        ['not a url/responses', 'not a url/responses'],
    ])('autofills malformed reverse proxy Responses URL variant %s', async (customURL, expectedURL) => {
        mocks.db.autofillRequestUrl = true

        const result = await requestOpenAIResponseAPI(baseArg({
            aiModel: 'reverse_proxy',
            customURL,
            previewBody: true,
            key: undefined,
        }))

        expect(result.type).toBe('success')
        expect(JSON.parse(result.result as string).url).toBe(expectedURL)
    })

    it('strips internal Responses continuation state from external request bodies', () => {
        const body:any = {
            model: 'gpt-5',
            store: false,
            input: [
                { id: 'rs_reasoning_bad', type: 'reasoning', content: [{ type: 'reasoning_text', text: 'private reasoning' }], summary: [] },
                { id: 'fc_bad', type: 'function_call', call_id: 'call_1', name: 'lookup', arguments: '{}', status: 'completed' },
            ],
            __lastOutput: [{ type: 'function_call', call_id: 'call_1' }],
        }
        const external = __testResponsesAPI.toExternalResponsesBody(body)

        body.input.push({ id: 'later_mutation', type: 'message', role: 'assistant', content: [] })

        expect(external).toEqual({
            model: 'gpt-5',
            store: false,
            input: [{ type: 'function_call', call_id: 'call_1', name: 'lookup', arguments: '{}', status: 'completed' }],
        })
        expect(JSON.stringify(external.input)).not.toContain('rs_reasoning_bad')
        expect(JSON.stringify(external.input)).not.toContain('private reasoning')
    })

    it('sanitizes non-streaming Responses tool continuation input with reasoning before a function call for store false', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'tool result' }] as any)
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: {
                    output: [
                        {
                            id: 'rs_reasoning_bad',
                            type: 'reasoning',
                            summary: [],
                        },
                        {
                            id: 'fc_allowed_server_id',
                            type: 'function_call',
                            call_id: 'call_lookup_1',
                            name: 'lookup',
                            arguments: '{"count":100,"offset":0}',
                            status: 'completed',
                        },
                    ],
                },
            })
            .mockResolvedValueOnce({
                ok: true,
                data: { output_text: 'final answer' },
            })

        const result = await requestOpenAIResponseAPI(baseArg({
            tools: [{ name: 'lookup', description: 'Lookup data', inputSchema: { type: 'object', properties: { count: { type: 'number' }, offset: { type: 'number' } } } }],
        }))

        expect(result).toEqual({ type: 'success', result: 'final answer' })
        expect(mocks.globalFetch).toHaveBeenCalledTimes(2)
        const followupBody = mocks.globalFetch.mock.calls[1][1].body
        const followupInputJSON = JSON.stringify(followupBody.input)
        expect(followupInputJSON).not.toContain('rs_reasoning_bad')
        expect(followupInputJSON).not.toContain('fc_allowed_server_id')
        expect(followupBody.input).toEqual(expect.arrayContaining([
            {
                type: 'function_call',
                call_id: 'call_lookup_1',
                name: 'lookup',
                arguments: '{"count":100,"offset":0}',
                status: 'completed',
            },
            {
                type: 'function_call_output',
                call_id: 'call_lookup_1',
                output: 'tool result',
            },
        ]))
    })

    it('recurses through a real-shaped Responses function-call fixture and preserves continuation input', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'fixture tool result' }] as any)
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: structuredClone(openAIResponsesToolCallFixture),
            })
            .mockResolvedValueOnce({
                ok: true,
                data: structuredClone(openAIResponsesSuccessFixture),
            })

        const result = await requestOpenAIResponseAPI(baseArg({
            tools: [{
                name: 'fixture_lookup',
                description: 'Lookup fixture data',
                inputSchema: {
                    type: 'object',
                    properties: { query: { type: 'string' } },
                    required: ['query'],
                    additionalProperties: false,
                },
            }],
        }))

        expect(result).toEqual({ type: 'success', result: 'response fixture ok' })
        expect(vi.mocked(callTool)).toHaveBeenCalledWith('fixture_lookup', { query: 'abc' })
        expect(mocks.globalFetch.mock.calls[0][1].body.tools).toEqual([expect.objectContaining({
            type: 'function',
            name: 'fixture_lookup',
        })])
        expect(openAIResponsesToolCallFixture).toMatchObject({
            output: [expect.objectContaining({
                type: 'function_call',
                id: 'fc_fixture_lookup',
                call_id: 'call_fixture_lookup',
                name: 'fixture_lookup',
                arguments: '{"query":"abc"}',
            })],
            tools: [expect.objectContaining({ type: 'function', name: 'fixture_lookup', strict: true })],
            usage: { total_tokens: 57 },
        })
        expect(mocks.globalFetch).toHaveBeenCalledTimes(2)
        const followupBody = mocks.globalFetch.mock.calls[1][1].body
        expect(followupBody.input).toEqual(expect.arrayContaining([
            {
                type: 'function_call',
                call_id: 'call_fixture_lookup',
                name: 'fixture_lookup',
                arguments: '{"query":"abc"}',
                status: 'completed',
            },
            {
                type: 'function_call_output',
                call_id: 'call_fixture_lookup',
                output: 'fixture tool result',
            },
        ]))
        expect(JSON.stringify(followupBody.input)).not.toContain('fc_fixture_lookup')
    })

    it('retries non-streaming tool follow-up before falling back to accumulated prefix', async () => {
        mocks.db.requestRetrys = 1
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'tool result' }] as any)
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: {
                    output_text: 'Need a lookup',
                    output: [{
                        type: 'function_call',
                        call_id: 'call_lookup_prefix',
                        name: 'lookup',
                        arguments: '{}',
                        status: 'completed',
                    }],
                },
            })
            .mockResolvedValueOnce({
                ok: false,
                data: { error: { message: 'follow-up failed' } },
            })
            .mockResolvedValueOnce({
                ok: false,
                data: { error: { message: 'follow-up failed again' } },
            })

        const result = await requestOpenAIResponseAPI(baseArg({
            tools: [{ name: 'lookup', description: 'Lookup data', inputSchema: { type: 'object' } }],
        }))

        expect(result).toEqual({ type: 'success', result: 'Need a lookup' })
        expect(mocks.globalFetch).toHaveBeenCalledTimes(3)
        expect(vi.mocked(alertError)).toHaveBeenCalledWith('Failed to fetch model response after tool execution')
    })

    it('propagates non-streaming tool follow-up failures when no prefix exists', async () => {
        mocks.db.simplifiedToolUse = true
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'tool result' }] as any)
        mocks.globalFetch
            .mockResolvedValueOnce({
                ok: true,
                data: {
                    output: [{
                        type: 'function_call',
                        call_id: 'call_lookup_no_prefix',
                        name: 'lookup',
                        arguments: '{}',
                        status: 'completed',
                    }],
                },
            })
            .mockResolvedValueOnce({
                ok: false,
                data: { error: { message: 'follow-up failed' } },
            })

        const result = await requestOpenAIResponseAPI(baseArg({
            tools: [{ name: 'lookup', description: 'Lookup data', inputSchema: { type: 'object' } }],
        }))

        expect(result).toEqual({ type: 'fail', result: 'HTTP follow-up failed' })
        expect(vi.mocked(alertError)).not.toHaveBeenCalled()
    })

    it('sanitizes streaming Responses tool continuation input for store false', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'stream tool result' }] as any)
        vi.mocked(fetchNative)
            .mockResolvedValueOnce({
                status: 200,
                headers: { get: () => 'text/event-stream' },
                body: sseStream([
                    'data: {"type":"response.completed","response":{"output_text":"Need a lookup","output":[{"id":"rs_stream_ignored","type":"function_call","call_id":"call_stream_1","name":"lookup","arguments":"{\\"query\\":\\"x\\"}","status":"completed"}]}}\n\n',
                ]),
            } as any)
            .mockResolvedValueOnce({
                status: 200,
                headers: { get: () => 'text/event-stream' },
                body: sseStream([
                    'data: {"type":"response.completed","response":{"output_text":"stream final","output":[{"type":"message","content":[{"type":"output_text","text":"stream final"}]}]}}\n\n',
                ]),
            } as any)

        const result = await requestOpenAIResponseAPI(baseArg({
            useStreaming: true,
            tools: [{ name: 'lookup', description: 'Lookup data', inputSchema: { type: 'object', properties: { query: { type: 'string' } } } }],
        }))

        expect(result.type).toBe('streaming')
        await collectStream(result.result as ReadableStream<Record<string, string>>)
        expect(vi.mocked(fetchNative)).toHaveBeenCalledTimes(2)
        const followupBody = JSON.parse(vi.mocked(fetchNative).mock.calls[1][1].body as string)
        expect(JSON.stringify(followupBody.input)).not.toContain('rs_stream_ignored')
        expect(followupBody.input).toEqual(expect.arrayContaining([
            expect.objectContaining({
                type: 'function_call',
                call_id: 'call_stream_1',
                name: 'lookup',
                arguments: '{"query":"x"}',
                status: 'completed',
            }),
            expect.objectContaining({
                type: 'function_call_output',
                call_id: 'call_stream_1',
                output: 'stream tool result',
            }),
        ]))
        expect(followupBody.input.find((item:any) => item.type === 'function_call')).not.toHaveProperty('id')
    })

    it('preserves streaming Responses assistant output in tool continuation input', async () => {
        vi.mocked(callTool).mockResolvedValueOnce([{ type: 'text', text: 'stream tool result' }] as any)
        vi.mocked(fetchNative)
            .mockResolvedValueOnce({
                status: 200,
                headers: { get: () => 'text/event-stream' },
                body: sseStream([
                    'data: {"type":"response.completed","response":{"output":[{"id":"msg_server_id","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Need a lookup","annotations":[]}]},{"id":"fc_server_id","type":"function_call","call_id":"call_stream_message","name":"lookup","arguments":"{\\"query\\":\\"x\\"}","status":"completed"}]}}\n\n',
                ]),
            } as any)
            .mockResolvedValueOnce({
                status: 200,
                headers: { get: () => 'text/event-stream' },
                body: sseStream([
                    'data: {"type":"response.completed","response":{"output_text":"stream final"}}\n\n',
                ]),
            } as any)

        const result = await requestOpenAIResponseAPI(baseArg({
            useStreaming: true,
            tools: [{ name: 'lookup', description: 'Lookup data', inputSchema: { type: 'object', properties: { query: { type: 'string' } } } }],
        }))

        expect(result.type).toBe('streaming')
        await collectStream(result.result as ReadableStream<Record<string, string>>)

        const followupBody = JSON.parse(vi.mocked(fetchNative).mock.calls[1][1].body as string)
        expect(followupBody.input).toEqual(expect.arrayContaining([
            {
                type: 'message',
                role: 'assistant',
                status: 'completed',
                content: [{ type: 'output_text', text: 'Need a lookup', annotations: [] }],
            },
            {
                type: 'function_call',
                call_id: 'call_stream_message',
                name: 'lookup',
                arguments: '{"query":"x"}',
                status: 'completed',
            },
            {
                type: 'function_call_output',
                call_id: 'call_stream_message',
                output: 'stream tool result',
            },
        ]))
        expect(JSON.stringify(followupBody.input)).not.toContain('msg_server_id')
        expect(JSON.stringify(followupBody.input)).not.toContain('fc_server_id')
    })

    it.each([
        ['IPv4 loopback', 'http://127.0.0.1:5000/v1/responses'],
        ['IPv6 loopback', 'http://[::1]:5000/v1/responses'],
        ['private IPv4', 'http://192.168.1.10:5000/v1/responses'],
        ['private IPv6', 'http://[fc00::1]:5000/v1/responses'],
        ['local hostname', 'http://model-server.local:5000/v1/responses'],
        ['single-label local hostname', 'http://litellm:5000/v1/responses'],
    ])('blocks browser local Responses streaming for %s before sending a request', async (_name, customURL) => {
        mocks.platform.isNodeServer = false
        mocks.platform.isTauri = false

        const result = await requestOpenAIResponseAPI(baseArg({
            customURL,
            useStreaming: true,
        }))

        expect(result).toEqual({
            type: 'fail',
            result: 'You are trying local request on streaming. this is not allowed dude to browser/os security policy. turn off streaming.',
        })
        expect(vi.mocked(fetchNative)).not.toHaveBeenCalled()
    })

    it('does not throw before fetch handling for malformed Responses streaming URLs', async () => {
        mocks.platform.isNodeServer = false
        mocks.platform.isTauri = false
        vi.mocked(fetchNative).mockResolvedValueOnce({
            status: 200,
            headers: { get: () => 'text/event-stream' },
            body: sseStream([
                'data: {"type":"response.completed","response":{"output_text":"malformed url fetched"}}\n\n',
            ]),
        } as any)

        const result = await requestOpenAIResponseAPI(baseArg({
            customURL: 'not a url',
            useStreaming: true,
        }))

        expect(result.type).toBe('streaming')
        expect(mocks.isLocalNetworkUrl).toHaveBeenCalledWith('not a url')
        expect(vi.mocked(fetchNative)).toHaveBeenCalledWith('not a url', expect.any(Object))
        await collectStream(result.result as ReadableStream<Record<string, string>>)
    })

    it('keeps Responses preview available before the browser local streaming guard', async () => {
        mocks.platform.isNodeServer = false
        mocks.platform.isTauri = false

        const result = await requestOpenAIResponseAPI(baseArg({
            customURL: 'http://localhost:5000/v1/responses',
            previewBody: true,
            useStreaming: true,
        }))

        expect(result.type).toBe('success')
        const preview = JSON.parse(result.result as string)
        expect(preview.url).toBe('http://localhost:5000/v1/responses')
        expect(preview.body.stream).toBe(true)
        expect(vi.mocked(fetchNative)).not.toHaveBeenCalled()
    })

    it.each([
        ['Node', true, false],
        ['Tauri', false, true],
    ])('allows local Responses streaming in %s mode', async (_name, isNodeServer, isTauri) => {
        mocks.platform.isNodeServer = isNodeServer
        mocks.platform.isTauri = isTauri
        vi.mocked(fetchNative).mockResolvedValueOnce({
            status: 200,
            headers: { get: () => 'text/event-stream' },
            body: sseStream([
                'data: {"type":"response.completed","response":{"output_text":"local ok"}}\n\n',
            ]),
        } as any)

        const result = await requestOpenAIResponseAPI(baseArg({
            customURL: 'http://0.0.0.0:5000/v1/responses',
            useStreaming: true,
        }))

        expect(result.type).toBe('streaming')
        expect(vi.mocked(fetchNative)).toHaveBeenCalledTimes(1)
        await collectStream(result.result as ReadableStream<Record<string, string>>)
    })

    it('accumulates text from real-shaped Responses streaming SSE events', async () => {
        vi.mocked(fetchNative).mockResolvedValueOnce({
            status: 200,
            headers: { get: () => 'text/event-stream' },
            body: sseStream([...openAIResponsesStreamingEventsFixture]),
        } as any)

        const result = await requestOpenAIResponseAPI(baseArg({ useStreaming: true }))

        expect(result.type).toBe('streaming')
        const chunks = await collectStream(result.result as ReadableStream<Record<string, string>>)
        expect(chunks.at(-1)?.['0']).toBe('response stream ok')
        expect(vi.mocked(fetchNative)).toHaveBeenCalledWith('https://api.openai.com/v1/responses', expect.objectContaining({
            body: expect.stringContaining('"stream":true'),
            interceptor: 'openai_response_api_streaming',
        }))
    })

    it('parses split CRLF SSE chunks, final unterminated events, text deltas, and function call deltas', async () => {
        const stream = __testResponsesAPI.getResponsesTranStream(baseArg())
        const chunksPromise = collectStream(stream.readable)
        const writer = stream.writable.getWriter()
        const encoder = new TextEncoder()

        await writer.write(encoder.encode('data: {"type":"response.output_text.delta","delta":"Hel"}\r\n\r'))
        await writer.write(encoder.encode('\ndata: {"type":"response.output_text.delta","delta":"lo"}\r\n\r\n'))
        await writer.write(encoder.encode('data: {"type":"response.function_call_arguments.delta","call_id":"call_1","delta":"{\\"q\\":"}\r\n\r\n'))
        await writer.write(encoder.encode('data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\\"q\\":\\"x\\"}","status":"completed"}}'))
        await writer.close()

        const chunks = await chunksPromise
        expect(chunks.at(-1)).toEqual({
            '0': 'Hello',
            __tool_calls: JSON.stringify({
                call_1: { type: 'function_call', call_id: 'call_1', name: 'lookup', arguments: '{"q":"x"}', status: 'completed' },
            }),
        })
    })

    it('does not duplicate text from a completed streaming event', async () => {
        const stream = __testResponsesAPI.getResponsesTranStream(baseArg())
        const chunksPromise = collectStream(stream.readable)
        const writer = stream.writable.getWriter()
        const encoder = new TextEncoder()

        await writer.write(encoder.encode('data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'))
        await writer.write(encoder.encode('data: {"type":"response.completed","response":{"output_text":"Hello","output":[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}]}}\n\n'))
        await writer.close()

        const chunks = await chunksPromise
        expect(chunks.at(-1)?.['0']).toBe('Hello')
    })

    it('streams reasoning_text deltas as thoughts', async () => {
        const stream = __testResponsesAPI.getResponsesTranStream(baseArg())
        const chunksPromise = collectStream(stream.readable)
        const writer = stream.writable.getWriter()
        const encoder = new TextEncoder()

        await writer.write(encoder.encode('data: {"type":"response.reasoning_text.delta","delta":"Thinking"}\n\n'))
        await writer.write(encoder.encode('data: {"type":"response.output_text.delta","delta":"Answer"}\n\n'))
        await writer.close()

        const chunks = await chunksPromise
        expect(chunks.at(-1)?.['0']).toBe('<Thoughts>\n\nThinking\n\n</Thoughts>\nAnswer')
    })

    it('uses completed streaming reasoning content without duplicating final output text', async () => {
        const stream = __testResponsesAPI.getResponsesTranStream(baseArg())
        const chunksPromise = collectStream(stream.readable)
        const writer = stream.writable.getWriter()
        const encoder = new TextEncoder()

        await writer.write(encoder.encode('data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'))
        await writer.write(encoder.encode('data: {"type":"response.completed","response":{"output":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"Reasoned once"}],"summary":[]},{"type":"message","content":[{"type":"output_text","text":"Hello"}]}]}}\n\n'))
        await writer.close()

        const chunks = await chunksPromise
        expect(chunks.at(-1)?.['0']).toBe('<Thoughts>\n\nReasoned once\n\n</Thoughts>\nHello')
    })

    it('emits useful text for streaming error events', async () => {
        const stream = __testResponsesAPI.getResponsesTranStream(baseArg())
        const chunksPromise = collectStream(stream.readable)
        const writer = stream.writable.getWriter()
        const encoder = new TextEncoder()

        await writer.write(encoder.encode('data: {"type":"response.failed","error":{"message":"stream failed"}}\n\n'))
        await writer.close()

        const chunks = await chunksPromise
        expect(chunks.at(-1)?.['0']).toContain('stream failed')
    })
})
