#pragma once

#include <stdio.h>
#include <string.h>

#include <fcntl.h> 
#include <unistd.h>

#include <termios.h>

#include <sys/ioctl.h>
#include <errno.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

class Uart{
	public:
		Uart()
		{
		}
		~Uart()
		{
			
		close(fd);

		}
		/**
		* 打开串口
		* 
		 */
		bool Open(const char * device, int _speed, int _parity, bool _should_block)
		{
			uart_path = device;
			speed = _speed;
			parity = _parity;
			should_block = _should_block;

			fd = open(uart_path, O_RDWR | O_NOCTTY | O_SYNC);
			if(-1==fd)
			{
				perror(uart_path);
				return false;
			}
			struct termios tty;
			memset (&tty, 0, sizeof tty);
			if (tcgetattr (fd, &tty) != 0)
			{
			       	printf("getattr error\n");
				return false;
			}
			cfsetospeed (&tty, speed);
			cfsetispeed (&tty, speed);

			tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
			// disable IGNBRK for mismatched speed tests; otherwise receive break
			// as \000 chars
			tty.c_iflag &= ~IGNBRK;         // disable break processing
			tty.c_lflag = 0;                // no signaling chars, no echo,
				                        // no canonical processing
			tty.c_oflag = 0;                // no remapping, no delays

				tcflush( fd , TCIOFLUSH );

				tty.c_cc[VMIN]  = should_block ? 23 : 0;// read doesn't block?
			tty.c_cc[VTIME] = 1;			// 0.5 seconds read timeout

			tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

			tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
				                        // enable reading
			tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
			tty.c_cflag |= parity;
			tty.c_cflag &= ~CSTOPB;
			tty.c_cflag &= ~CRTSCTS;

			if (tcsetattr (fd, TCSANOW, &tty) != 0)
			{
			  	printf("setattr error\n");
					return false;
			}
			return true;
		}

		/**
		* 重启串口
		* 
		 */
		bool restart()
		{
		  
		   Close();
		   fd = open(uart_path, O_RDWR | O_NOCTTY | O_SYNC);
		    struct termios tty;
		    memset (&tty, 0, sizeof tty);
		    if (tcgetattr (fd, &tty) != 0)
		    {
			   	printf("getattr error\n");
				return false;
		    }
		    cfsetospeed (&tty, speed);
		    cfsetispeed (&tty, speed);

		    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
		    // disable IGNBRK for mismatched speed tests; otherwise receive break
		    // as \000 chars
		    tty.c_iflag &= ~IGNBRK;         // disable break processing
		    tty.c_lflag = 0;                // no signaling chars, no echo,
				                        // no canonical processing
		    tty.c_oflag = 0;                // no remapping, no delays

			tcflush( fd , TCIOFLUSH );

			tty.c_cc[VMIN]  = should_block ? 23 : 0;// read doesn't block?
		    tty.c_cc[VTIME] = 1;			// 0.5 seconds read timeout

		    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

		    tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
				                        // enable reading
		    tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
		    tty.c_cflag |= parity;
		    tty.c_cflag &= ~CSTOPB;
		    tty.c_cflag &= ~CRTSCTS;

		    if (tcsetattr (fd, TCSANOW, &tty) != 0)
		    {
			  printf("setattr error\n");
					return false;
		    }
			return true;
	
		}
		
		
		/**
		* 读取串口数据
		* 
		 */
		bool ReadData(unsigned char data[23], unsigned char & mode, float & yaw, short & pitch, float & timestamp, short & heat42, short & heat17, unsigned char & bulletSpeed17)
		{
			int readsize = read( fd , data , 23 );
			tcflush( fd , TCIFLUSH );
			if (readsize != 23) {
				//cout << "READ ERROR:" << readsize << endl;
				return false;
			}
	
	
	
			union FloatToCharUnion {
				float num;
				unsigned char data[4];
			}_FloatToCharUnion;

			union Int16ToCharUnion {
				short num;
				unsigned char data[2];
			}_Int16ToCharUnion;

			mode = data[2];

			_FloatToCharUnion.data[0] = data[3];
			_FloatToCharUnion.data[1] = data[4];
			_FloatToCharUnion.data[2] = data[5];
			_FloatToCharUnion.data[3] = data[6];
			yaw = _FloatToCharUnion.num;

			_Int16ToCharUnion.data[0] = data[7];
			_Int16ToCharUnion.data[1] = data[8];
			pitch = _Int16ToCharUnion.num;

			_FloatToCharUnion.data[0] = data[9];
			_FloatToCharUnion.data[1] = data[10];
			_FloatToCharUnion.data[2] = data[11];
			_FloatToCharUnion.data[3] = data[12];
			timestamp = _FloatToCharUnion.num / 10e5;

			_Int16ToCharUnion.data[0] = data[13];
			_Int16ToCharUnion.data[1] = data[14];
			heat42 = _Int16ToCharUnion.num;

			_Int16ToCharUnion.data[0] = data[15];
			_Int16ToCharUnion.data[1] = data[16];
			heat17 = _Int16ToCharUnion.num;

			bulletSpeed17 = data[17];

			unsigned char check = 0;
			for (int inc = 0; inc < 22; inc++) {
				check ^= data[inc];
			}
	
			if (check == data[22] && data[0] == 0x2d && data[1] == 0xd2 )
				return true;
			else
				return false;
		}
		
		/**
		* 发送数据
		* 
		 */
		bool SendData(char* data,int len)
		{
	
	
		if( write ( fd , data , len ) != len ){
			tcflush( fd , TCOFLUSH );
			//printf("uart send error");
			return false;   // send 22 character greeting
			}
	
		/*for(int i = 0;i < 16; ++i)
			cout << (int)(unsigned char)data[i] << " ";
		cout << endl;*/
	
		usleep ((len) * 100);
		tcflush( fd , TCOFLUSH );
		return true;
	
		}

		/**
		*  发送数据帧
		* 
		 */
		bool send(float yaw, float pitch, float distance)
		{
			char data[16];
			union FloatToCharUnion {
				float num;
				char data[4];
			}_FloatToCharUnion;
			union ShortToCharUnion {
				short num;
				char data[2];	
		
			}_ShortToCharUnion;

			data[0] = 0x3d;
			data[1] = 0xd3;

			//data[2] = 6;

			/*if (shoot42)
				data[3] = 1;
			else
				data[3] = 0;

			if (shoot17)
				data[4] = 1;
			else
				data[4] = 0;*/

			_FloatToCharUnion.num = yaw;
			data[2] = _FloatToCharUnion.data[0];
			data[3] = _FloatToCharUnion.data[1];
			data[4] = _FloatToCharUnion.data[2];
			data[5] = _FloatToCharUnion.data[3];

			_FloatToCharUnion.num = pitch;
			data[6] = _FloatToCharUnion.data[0];
			data[7] = _FloatToCharUnion.data[1];
			data[8] = _FloatToCharUnion.data[2];
			data[9] = _FloatToCharUnion.data[3];

			_FloatToCharUnion.num = distance;
			data[10] = _FloatToCharUnion.data[0];
			data[11] = _FloatToCharUnion.data[1];
			data[12] = _FloatToCharUnion.data[2];
			data[13] = _FloatToCharUnion.data[3];

			data[14] = 0;
			data[15] = 0;

			for (int inc = 0; inc < 15; inc++)
			{
				data[15] ^= data[inc];
			}

			//for (int i = 0; i < 14; i++) {
			//	cout << (int)(unsigned char)data[i] << " ";
			//}

			if (SendData(data, 16) == false) {
				//cout << "send filed" << endl;
				return false;
			}
			else 
				return true;
		}

		bool send_ttltest(float senddata)//USBTLL single data test with connecting to personal pc
		{
			char data[4];
			union FloatToCharUnion {
				float num;
				char data[4];
			}_FloatToCharUnion;

			_FloatToCharUnion.num = senddata;
			data[0] = _FloatToCharUnion.data[0];
			data[1] = _FloatToCharUnion.data[1];
			data[2] = _FloatToCharUnion.data[2];
			data[3] = _FloatToCharUnion.data[3];
			for (int inc = 0; inc < 3; inc++)
			{
				data[4] ^= data[inc];
			}
			if (SendData(data, 4) == false) {
				//cout << "send filed" << endl;
				return false;
			}
			else 
				return true;
		}


		/**
		* 这个发送空数据，目前还不知道用来做什么
		* 
		 */
		bool SendMiss()
		{
			char _data[16];
	
			for(int i = 0; i < 16 ; ++i)
				_data[i] = i;

			_data[0] = 0x3d;
			_data[1] = 0xd3;
			_data[2] = 0;
			_data[3] = 0;
			_data[4] = 0;
	
	
			_data[15] = 0;
	
			for (int inc = 0; inc < 15; inc++){
				_data[15] ^= _data[inc];
			}
	
			if (SendData(_data, 16)){
				return true;
			}
			else {
				return false;
			}
		}


		/**
		* 关闭窗口
		* 
		 */
		void Close()
		{
		   close(fd);
		}
	private:
		const char *uart_path;
		int fd;
		int speed;
		int parity;
		bool should_block;		

};
