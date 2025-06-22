import os
from openai import OpenAI


# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Openai客户端，从环境变量中读取您的API Key



# def get_answer(question: str):
#     client = OpenAI(
#         # 此为默认路径，您可根据业务所在地域进行配置
#         base_url="https://ark.cn-beijing.volces.com/api/v3",
#         # 从环境变量中获取您的 API Key
#         api_key="e864c037-480f-4533-bb04-df290365997f",
#     )
#
#     # Non-streaming:
#     print("----- standard request -----")
#     completion = client.chat.completions.create(
#         # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
#         model="doubao-lite-4k-character-240828",
#         messages=[
#             {"role": "system", "content": "你是人工智能助手"},
#             {"role": "user", "content": question},
#         ],
#     )
#     print(f"回答：\n{completion.choices[0].message.content}")
#     return completion.choices[0].message.content
#申请网址：https://exp.volcengine.com/ark?model=doubao-lite-4k-character-240828
def get_answer(question: str):
    try:
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="e864c037-480f-4533-bb04-df290365997f",
        )
        completion = client.chat.completions.create(
            model="doubao-lite-4k-character-240828",
            messages=[
                {"role": "system", "content": "你是教学小助手"},
                {"role": "user", "content": question},
            ],
        )
        # print(f"回答：\n{completion.choices[0].message.content}")
        return completion.choices[0].message.content
    except Exception as e:
        print(f"大模型调用出错: {e}")
        return "抱歉，大模型调用出错，请稍后再试。"



if __name__ == "__main__":
    print("调用大模型获取答案...")
    get_answer("你是谁")


# Streaming:
# print("----- streaming request -----")
# stream = client.chat.completions.create(
#     # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
#     model="doubao-lite-4k-character-240828",
#     messages=[
#         {"role": "system", "content": "你是人工智能助手"},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
#     # 响应内容是否流式返回
#     stream=True,
# )
# for chunk in stream:
#     if not chunk.choices:
#         continue
#     print(chunk.choices[0].delta.content, end="")
# print()