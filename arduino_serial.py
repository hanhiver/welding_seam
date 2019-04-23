import serial


class arduino_serial:
    STRGLO = "" # 读取的数据
    serialPort = "" # 串口号
    baudrate = 9600 # 波特率
    timeout = 0.5 # 超时设置
    ser = None

    def __init__(self, serialPort, baudrate = baudrate, timeout = timeout):
        self.serialPort = serialPort
        self.baudrate = baudrate
        self.timeout = timeout

    # 打开串口
    # 端口、波特率、超时设置
    def openPort(self):
        ret = False

        try:
            ser = serial.Serial(self.serialPort, self.baudrate, timeout=self.timeout)
            if(ser.is_open):
                self.ser = ser
                ret = True

        except Exception as e:
            print("---Exception---", e)
    
        return ret
    
  
    # 关闭串口
    def closePort(self):
        self.ser.close()

    # 写数据
    def writePort(self,msg):
        input_s = msg.strip()
        send_list = []

        while(input_s != ''):
            try:
                num = int(input_s[0:2], 16)
            except ValueError:
                return None

            input_s = input_s[2:].strip()
            send_list.append(num)
        
        input_s = bytes(send_list)
        self.ser.write(input_s)
        
        return 1

if __name__ == '__main__':
    Obj = arduino_serial('/dev/ttyUSB0')
    ret = Obj.openPort()
    if(ret == True):
        Obj.writePort('E701450124005A0008004EFE')

    Obj.closePort()
