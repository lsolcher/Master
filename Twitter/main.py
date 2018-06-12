from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import API

ckey = 'rGxOFKgKRoGo1Kpl1FEqjNGlI'
csecret = 'nnk4mqbRdOQQsCy8rIwCAxHnFUO6iGgjkpSsM96bGSZcANg7mR'
atoken = '912478974-s9PU6WjEeZC0olc57moslUByXX7UeXxttrPH7YbK'
asecret = 'Vy5hWwthlxuU6qSVoq91Bb4TjfJo9sHSrmx66BN04zoTX'


class Listener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

api = API(auth)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track=["AFD"])


test = api.lookup_users(user_ids=['17006157','59145948','157009365'])
for user in test:
    print (user.screen_name)
    print (user.name)
    #print user.description
    #print user.followers_count
    #print user.statuses_count
    #print user.url
#print(user)
