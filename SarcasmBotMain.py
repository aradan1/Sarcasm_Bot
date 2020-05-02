
# coding: utf-8
import classifier
from pickle import dump, load # to save model so we dont waste 2-3 mins every time we restart


import asyncio as aio
import telepot

from telepot.aio.loop import MessageLoop
from telepot.aio.delegate import pave_event_space, per_chat_id, create_open, include_callback_query_chat_id



class SarcasmBot(object):
    
    instance = None
    
    def __init__(self):
        assert SarcasmBot.instance is None
        SarcasmBot.instance = self
        
        self.bot = None
        self.loop = aio.get_event_loop()
        self.msg_loop = None
        
        
    def start(self, token):
        self.bot = telepot.aio.DelegatorBot(token, [
            include_callback_query_chat_id(
                pave_event_space())(
                per_chat_id(), create_open, SarcasmUser, timeout=10),
            ])
        
        self.msg_loop = MessageLoop(self.bot)
        self.loop.create_task(MessageLoop(self.bot).run_forever())
        self.loop.run_forever()
        
    def stop(self):
        SarcasmBot.instance = None
        if self.msg_loop:
            self.msg_loop.cancel()
            self.msg_loop = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        self.stop()



        
class SarcasmUser(telepot.aio.helper.ChatHandler):
    
    def __init__(self, *args, **kwargs):
        super(SarcasmUser, self).__init__(*args, **kwargs)
        print('Created {}'.format(self.id))
       
    
    async def on_chat_message(self, msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        
        if 'text' in msg:
            if msg['text'] == '/help':
                await self.sender.sendMessage("Send a message. If it starts with a vocal (a,e,i,o,u) it will show an emoji, else it will show a clown emoji")
                return
            
            message = msg['text']
            print(message)
            
            prediction = predictor.predict([message])[0]


            if prediction:
                # is sarcastic
                await self.sender.sendMessage("🤡")
            else:
                # isn't sarcastic
                await self.sender.sendMessage("👌")

                
        else:
            await self.sender.sendMessage("Message must be a text")
            
    async def on_close(self, ex):
        print('Closed {}'.format(self.id))



if __name__ == '__main__':


    try:
        with open('data/predictor.pkl', 'rb') as f:
            predictor = load(f)

    except IOError:
        
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        path = askopenfilename()

        df = classifier.readDf(path)
        x_train, x_test, y_train, y_test = train_test_split(df['comment'], df['label'], random_state=10)

        predictor = classifier.modelFitting(x_train, x_test, y_train, y_test)
        dump(predictor, open('data/predictor.pkl', 'wb'))
    

    print("Bot started") 
    # Se crea un bot i inicia
    with SarcasmBot() as bot:

        # Start bot
        bot.start(open('TOKEN').read().strip())