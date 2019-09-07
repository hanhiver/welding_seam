"""
General Logging module. 

This module provide a logger manager across threads and processes. 

HAN Dong 
"""

import sys
import multiprocessing
import logging
import logging.handlers
import socket

import time

LOG_LEVELS = {
	'debug' : logging.DEBUG, 
	'info' : logging.INFO, 
	'warning' : logging.WARNING, 
	'error' : logging.ERROR, 
	'critical' : logging.CRITICAL,
}

"""
Specify a keyword such like 'debug', the function automatically return logging leve. 
If the keyword cannot found in LOG_LEVELS, simply return logging.NOSET. 

level_name:
	string of logging level. such like 'debug', 'info', etc. 

return: 
	return the official logging leve. such like logging.DEBUG, etc. 
	if the string is not predefined, return logging.NOSET. 
"""
def get_level(level_name):
	level = LOG_LEVELS.get(level_name, logging.NOTSET)
	return level 

############################
### Local Logging system ###
############################

class LoggerManager():
	"""
	Init func of the LoggerManager()

	filename:
		Name of the log file that will generated. 
		The log file will rotated automatically by adding .1, .2, etc. 

	log_level: 
		Default log level. The message lower then the log level will be ignore. 

	max_bytes: 
		Max size of the log file. After that, the file will be rotated. 

	backup_count: 
		Max number of log files will be saved in the system. For the older log files, 
		system will delete them automatically.  
	"""
	def __init__(self, filename = None, 
				 log_level = 'warning', # Default log level is DEBUG
				 max_bytes = 50000000, # 50MB Log file by default. 
				 backup_count = 5, # Save 5 log files by default. 
				 log_format = None):

		self.log_level = get_level(log_level)

		hostname = socket.gethostname().split('.')[0]
		if not filename:
			filename = sys.argv[0].split('.')[0]

		# Add hostname as the suffix to log files.
		filename = filename + '.log.' + hostname

		# Add Manager.Queue() to hanlde the multi-processes communication. 
		self.manager = multiprocessing.Manager()
		self.queue = self.manager.Queue()

		# Standard loging message format:
		# time, log_level, <process_id:thread_name> - [module.func_name] log_messages. 
		if not log_format:
			log_format = '%(asctime)s %(levelname)7s  <%(process)s:%(threadName)s> - [%(name)s.%(funcName)s] - %(filename)s[line:%(lineno)d] %(message)s'
		
		self.formatter = logging.Formatter(log_format)

		# Queue handler to recieve msg from other processes. 
		self.queue_handler = logging.handlers.QueueHandler(self.queue)
		self.queue_handler.setFormatter(self.formatter)

		# File handler to save/rotate the log files. 
		self.file_handler = logging.handlers.RotatingFileHandler(
							filename, maxBytes = max_bytes, backupCount = backup_count)
		self.file_handler.setFormatter(self.formatter)

		# Stream handler that output the messages to the standard output. 
		self.stream_handler = logging.StreamHandler()
		self.stream_handler.setFormatter(self.formatter)

		# Log listener thread to write the log files. 
		self.listener = logging.handlers.QueueListener(self.queue, self.file_handler)
		
		# Start the log listener thread. 
		self.listener.start()

	"""
	Stop the listener thread. 
	"""
	def stop(self):
		self.listener.stop()

	"""
	Return a logger that can output log messages. 

	module: 
		Module name that will be shown in the log messages. 
	"""
	def get_logger(self, module = ''):

		# Get a default Logger. 
		new_logger = logging.getLogger(module)

		# Set the log level. 
		new_logger.setLevel(self.log_level)

		# Associate the logger to file_handler and stream_handler. 
		new_logger.addHandler(self.queue_handler)
		new_logger.addHandler(self.stream_handler)

		return new_logger


###############
### Example ###
###############

"""
Example to use the loggerManager module. 

This will generate log file and log messages from:
1. Main Process. 
2. Chiled Thread. 
3. Chiled Process. 
"""

logger_manager = None

def write_log():
	# Involve global LoggerManager. 
	global logger_manager

	# Get logger inside thread/processes. 
	logger = logger_manager.get_logger('WriteLogFunc')
	logger.debug('A DEBUG infomation in thread. ')
	logger.info('A INFO information in thread. ')

def main():
	global logger_manager

	# Initilaize a LoggerManager
	logger_manager = LoggerManager()

	# Get logger in main() function. 
	logger = logger_manager.get_logger('Mainfunc')
	logger.debug('A DEBUG information in Main. ')
	logger.info('A INFO information in Main. ')

	from threading import Thread 
	t = Thread(target = write_log)
	t.start()
	t.join()

	p = multiprocessing.Process(target = write_log)
	p.start()
	p.join()

	# Stop the logger listener thread. (Optional)
	logger_manager.stop()

if __name__ == '__main__':
	main()











				

