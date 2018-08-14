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

    def gauss_answer(self, question):
        session_id = self.get_session_id()

        message = {
            "header":{
            "querystring": {"appId": str(self.appid), "type": "dingding_channel"},
            "memberType": "guest", "imsid": session_id,
            "appName": "alixm", "chatSequence": 1, "msglog": 3, "sessionId": "",
            "bizType": "", "showType": "",
            "robotCode": str(self.robot_code), "qType": "text",
            "encryptedUserId": "", "tenantId": 3}, "type": "text",
            "subType": "text", "body": question,
            "fromId": 8610525421470624, "fromName": "游客"
        }
        payload = {
            "appId": str(self.appid),
            "auth_vendor": "ddg",
            "access_token": self.token,
            "_input_charset": "utf-8",
            "message": json.dumps(message),
        }
        url = "http://pre-beebot.alibaba.com/base/pivot?"
        rsp = urlopen(url + urlencode(payload))
        json_rst = json.loads(rsp.read())
        if self.print_uuid:
            print(self.chat_uuid(json_rst['data'][1]))
        if 'data' not in json_rst:
            return ''
        return self.parse_answer(json_rst['data'][1])

    def chat_uuid(self, json_answer):
        return json_answer['header']['chatUuid']

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
    api = ChatApi(1011986, '1011986_gausscode_10159')

    f_api = open('query_A.log.MY','r+',encoding='utf-8')
    previous = f_api.readlines()[-1].split()[0]
    #f_api.write('log_ans_type\tlog_ans\n')
    for index,line in enumerate(open('D:/queryrewrite/inputs/query_A', 'r', encoding='utf-8')):
        line = line.strip()
        if index<=int(previous):
            continue
        #line = 'my order has been cancelled..how can i claim my refund?'
        ans = api.gauss_answer(line)
        print('%d\t%s\t%s' % (index,ans.answer_type, ans.answer_content))
        f_api.write('\n%d\t%s\t%s' % (index,ans.answer_type, ans.answer_content))
    f_api.close()