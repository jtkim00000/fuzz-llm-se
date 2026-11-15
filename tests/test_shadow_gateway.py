import json
import unittest
import httpx

from gateway.app import Worker, create_app


class Stream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
        yield b'data: [DONE]\n\n'


class ShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_cannot_change_route_or_break_pinning(self):
        async def provider(request, body):
            # Missing telemetry is a rejected proposal; actual baseline remains.
            return {}, {}

        calls=[]
        def handler(request):
            calls.append(request.url.host)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Stream())

        app=create_app([Worker('bf16','http://one'),Worker('fp8','http://two')],
            transport=httpx.MockTransport(handler),policy_provider=provider)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
                with self.assertLogs('gateway',level='INFO') as logs:
                    for session in ['a','b','a']:
                        r=await client.post('/v1/chat/completions',headers={'X-Session-ID':session},
                                            json={'stream':True,'messages':[{'role':'user','content':'PRIVATE_BODY'}]})
                        self.assertEqual(r.status_code,200)
                        self.assertIn('[DONE]',r.text)
                events=[json.loads(line.split(':',2)[-1]) for line in logs.output]
                self.assertEqual(len([e for e in events if e['event']=='shadow_decision']),2)
                self.assertNotIn('PRIVATE_BODY',''.join(logs.output))
        self.assertEqual(calls,['one','two','one'])

    async def test_provider_failure_preserves_baseline(self):
        async def provider(*args): raise ValueError('sensitive exception')
        app=create_app([Worker('bf16','http://one')],policy_provider=provider,
            transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=Stream())))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
                r=await client.post('/v1/chat/completions',headers={'X-Session-ID':'a'},json={})
                self.assertEqual(r.status_code,200)


if __name__=='__main__': unittest.main()
