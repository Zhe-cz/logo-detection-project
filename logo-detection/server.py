import tornado
from tornado.options import define, options
import tornado.ioloop
import tornado.options
import tornado.httpserver
import tornado.web
import os, json
from yolo import YOLO
from PIL import Image
from io import BytesIO
import base64

yolo_net = YOLO()
define("port", default=8000, help="run on the given port", type=int)




class Index_number(tornado.web.RequestHandler):
	def get(self):
		# self.write(json.dumps(TARGET_DATAS["list"]))
		self.render("index.html")
	def post(self):
		images = Image.open(BytesIO(base64.b64decode(self.get_argument("img").split("base64,")[1])))
		im, img = yolo_net.detecter_images(images)
		images = base64.b64encode(im.getvalue()).decode()
		self.write(images)


def main():
	tornado.options.parse_command_line()
	html_servers = tornado.httpserver.HTTPServer(
		tornado.web.Application(
			handlers=((r"/", Index_number),),
			template_path=os.path.join(os.path.dirname(__file__), "templates"),
			static_path=os.path.join(os.path.dirname(__file__), "statics"),
			debug=True,
		))
	print("Click to enter http://localhost:%s" % options.port)
	print("sign out Ctrl + C")
	html_servers.listen(options.port)
	tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
	main()
