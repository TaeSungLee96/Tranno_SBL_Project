
from slack_cleaner2 import *

token_key ='xoxp-1983717984807-2004655107572-2099716588228-31dcf982817d83b2044935c55e903364'
s = SlackCleaner(token_key)
# list of users
s.users
# list of all kind of channels
s.conversations

# delete all messages in general channels
for msg in s.msgs(filter(match('fdd_alarm_service'), s.conversations)): #채널명 match함수에 넣기
  # delete messages, its files, and all its replies (thread)
  msg.delete(replies=True, files=True)

# delete all general messages and also iterate over all replies
for msg in s.c.general.msgs(with_replies=True):
  msg.delete()

