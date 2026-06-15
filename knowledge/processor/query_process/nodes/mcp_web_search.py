import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode


class MCPWebSearchNode(BaseNode):


    # 外层同步process
    def process(self, state):
        return asyncio.run(self._async_process(state))


    # 内部异步调用
    async def _async_process(self, state):
        # 参数校验
        destinations, rewritten_query = self._validate_params(state)
        # 调用mcp工具
        mcp_result = await self._search_by_mcp(rewritten_query)
        if not mcp_result:
            return {"web_search_docs": []}
        return {"web_search_docs": mcp_result}


    def _validate_params(self, state):
        # 兼容新的意图识别架构，替换原有的 item_names
        destinations = state.get("destinations", [])
        rewritten_query = state.get("rewritten_query")

        if not rewritten_query:
            raise ValueError("rewritten_query is required for web search")

        return destinations, rewritten_query


    async def _search_by_mcp(self, rewritten_query):
        # 创建mcp服务客户端
        mcp_client = None

        # 建立连接需要header信息
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json"
        }

        try:
            mcp_client = MCPServerStreamableHttp(
                name="联网搜索",
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": headers,
                },
                cache_tools_list=True
            )

            await mcp_client.connect()
            # 调用mcp_client
            mcp_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": rewritten_query,
                    "count": 3
                }
            )
            if not mcp_result:
                return []

            context = mcp_result.content[0].text
            # 反序列化
            context_dict = json.loads(context)
            pages = context_dict.get("pages", [])

            # 从pages列表获取具体数据，snippet答案 title标题问题 url网页地址
            # 列表 封装最终数据
            search_result = []
            for page in pages:
                snippet = page.get('snippet', "")
                title = page.get('title', "")
                url = page.get('url', "")
                search_result.append({
                    "snippet": snippet,
                    "title": title,
                    "url": url
                })
            return search_result
        except Exception as e:
            self.logger.error(f"MCP联网搜索发生异常: {e}")
            raise e
        finally:
            if mcp_client:
                await mcp_client.cleanup()

if __name__ == "__main__":
    state = {
        "destinations": ['成都'],
        "intents": ['交通指南'],
        "rewritten_query": "如何从成都前往春熙路？",
        "confirmed_docs": [],
    }

    web_search = MCPWebSearchNode()
    result = web_search.process(state)
    print(result)