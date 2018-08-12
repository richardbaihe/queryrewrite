#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re, json, datetime, urllib, time
import sys
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import quote, urlencode
import json
import re


def parse_jsonp(jsonp_str):
    try:
        return re.search('^[^(]*?\((.*)\)[^)]*$', jsonp_str).group(1)
    except:
        raise ValueError('Invalid JSONP')


class RobotAnswer(object):
    def __init__(self, answer_type, answer_content):
        self.answer_type = answer_type
        self.answer_content = answer_content

    def __str__(self):
        return "#{}# #{}#".format(self.answer_type, self.answer_content)

    def __eq__(self, other):
        if self.answer_type == 'rec' and self.answer_type == other.answer_type:
            if set(self.answer_content.split('|')) \
                    == set(other.answer_content.split()):
                return True
        # no answer compare
        if self.answer_type == other.answer_type and\
                self.answer_type in ('noanswer', 'chat'):
            return True
        if self.answer_type == other.answer_type and \
                self.answer_content == other.answer_content:
            return True
        return False

    @classmethod
    def load(cls, line):
        if line == '':
            return cls('no_answer', [])
        pattern = re.compile(r'#([^#]*)# #([^#]*)#')
        matcher = pattern.search(line)
        answer_type = matcher.group(1)
        answer_content = matcher.group(2)
        return cls(answer_type, answer_content)


class ChatApi(object):
    def __init__(self, appid, robot_code, internal=False,
                 token_id=None, print_uuid=False):
        '''
        internal: 是否是内部机器人
        '''
        self.appid = appid
        self.robot_code = robot_code
        self.internal = internal
        if token_id is None:
            self.token = self.get_token()
        else:
            self.token = token_id
        self.print_uuid = print_uuid

    def get_token(self,):
        # 获取对外的token
        token_url = "https://beebot.alibaba.com/auth/get_access_token?" \
                    "auth_vendor=ddg&expiresIn=86400&_input_charset=utf-8"
        rst = json.loads(urlopen(token_url).read().decode('utf8'))
        return rst['data']

    def gauss_answer(self, question,token_id):
        session_id = self.get_session_id()

        message = {
            "header":{
            "querystring": {"env":"prod","token":token_id,
            "appId":"1011986","lang":"null","from":"dd"},
            "memberType": "normal", "imsid": session_id,
            "appName": "alixm", "chatSequence": 1, "msglog": 3, "sessionId": "",
            "bizType": "", "showType": "",
            "robotCode": str(self.robot_code), "qType": "text",
            "encryptedUserId": "", "tenantId": 3}, "type": "text",
            "subType": "text", "body": question,
            "fromId": 4046796941, "fromName": "白赫"
        }
        payload = {
            "appId": str(self.appid),
            "auth_vendor": "dda",
            "access_token": token_id,
            "_input_charset": "utf-8",
            "message": json.dumps(message),
        }
        url = "https://console-beebot.alibaba.com/robot/pivot?"
        #rsp = urlopen('https://console-beebot.alibaba.com/robot/pivot?message=%7B%22header%22%3A%7B%22querystring%22%3A%7B%22env%22%3A%22prod%22%2C%22token%22%3A%228ae027bc9a0148e169cf4f950904c019dd56ff57e7be366c50426c077b868620a0e8b3634ded568a29d503824efac1008f8c8fdb8cd520602716ee6687c184358ca1bcecdab929550a1d00b6231956a7e6f190a48627b3648ff5ecad4fd4bcf8c823c747e227884a35c1f78311da65fbab39e082e8c5ea12628a28bd364acbab%22%2C%22appId%22%3A%221011986%22%2C%22lang%22%3A%22null%22%2C%22from%22%3A%22dd%22%7D%2C%22memberType%22%3A%22normal%22%2C%22imsid%22%3A%229fe2fe8f6dab4f7caf9e5aaaf4a17208%22%2C%22appName%22%3A%22alixm%22%2C%22chatSequence%22%3A1%2C%22msglog%22%3A3%2C%22sessionId%22%3A%22%22%2C%22bizType%22%3Anull%2C%22showType%22%3A%22%22%2C%22robotCode%22%3A%221011986_gausscode_10159%22%2C%22qType%22%3A%22text%22%2C%22encryptedUserId%22%3A%22%22%2C%22tenantId%22%3A3%7D%2C%22type%22%3A%22text%22%2C%22subType%22%3A%22text%22%2C%22body%22%3A%22Order+Status%22%2C%22fromId%22%3A4046796941%2C%22fromName%22%3A%22%E7%99%BD%E8%B5%AB%22%7D&access_token=8ae027bc9a0148e169cf4f950904c019dd56ff57e7be366c50426c077b868620a0e8b3634ded568a29d503824efac1008f8c8fdb8cd520602716ee6687c184358ca1bcecdab929550a1d00b6231956a7e6f190a48627b3648ff5ecad4fd4bcf8c823c747e227884a35c1f78311da65fbab39e082e8c5ea12628a28bd364acbab&auth_vendor=dda&appId=1011986&_input_charset=utf-8')
        rsp = urlopen(url + urlencode(payload))
        json_rst = json.loads(rsp.read())
        if self.print_uuid:
            print(self.chat_uuid(json_rst['data'][1]))
        if 'data' not in json_rst:
            return ''
        return self.parse_answer(json_rst['data'][1])

    def chat_uuid(self, json_answer):
        return json_answer['hea' \
                           'der']['chatUuid']

    def parse_answer(self, json_answer):
        type, body = json_answer['type'], json_answer['body']
        if type == 'live_agent_event':
            return RobotAnswer("bfw", body)
        else:
            body = json.loads(json_answer['body'])
        if type == 'robot' or type == 'knowledge':
            if 'answerSource' in body and body['answerSource'] == 'CHAT':
                answer = RobotAnswer("chitchat", body['text'])
            elif 'answerSource' in body and body['answerSource'] == 'BOT_ANSWER':
                answer = RobotAnswer("bfw", body['text'])
            elif 'knowledge_title' in body:
                # 直出
                answer = RobotAnswer("kbs", body['knowledge_title'])
            else:
                answer = RobotAnswer(type, json_answer['body'])
        elif type == 'recommend':
            # 推荐
            answer = body['recommends'][0]['title']
            answer = RobotAnswer("kbs", answer)
        # elif type == 'robot':
        #     answer = RobotAnswer("condition", body['text'])
        # no answer
        else:
            answer = RobotAnswer(type, json_answer['body'])
        return answer

    def get_session_id(self, ):
        return 'rank1102' + str(datetime.datetime.now()).replace(' ', '')

    #
    # def get_baseline_result():
    #

if __name__ == '__main__':
    # api = ChatApi(1000362, 'app_code_gausscode_70', True)
    token = '22d8daa7f3d457c92c2543aa72996fb0dd56ff57e7be366c50426c077b868620a0e8b3634ded568a29d503824efac1008f8c8fdb8cd520602716ee6687c184358ca1bcecdab929550a1d00b6231956a7e6f190a48627b3648ff5ecad4fd4bcf8c823c747e227884a35c1f78311da65fbab39e082e8c5ea12628a28bd364acbab'
    api = ChatApi(1011986, '1011986_gausscode_10159')

    f_api = open('query_A.log.MY','r+',encoding='utf-8')
    previous = f_api.readlines()[-1].split()[0]
    #f_api.write('log_ans_type\tlog_ans\n')
     for index,line in enumerate(open('D:/queryrewrite/inputs/query_A', 'r', encoding='utf-8')):
        line = line.strip()
        if index<=int(previous):
            continue
        #line = 'my order has been cancelled..how can i claim my refund?'
        ans = api.gauss_answer(line, token_id=token)
        print('%d\t%s\t%s' % (index,ans.answer_type, ans.answer_content))
        f_api.write('\n%d\t%s\t%s' % (index,ans.answer_type, ans.answer_content))
    f_api.close()