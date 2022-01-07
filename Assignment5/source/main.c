#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

#define IRQ_NUM  1  //newly add

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2

void *dma_buf;
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdevp;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);


// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}


static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	
    // Print the answer
    unsigned int answer = myini(DMAANSADDR);
	unsigned int is_readable = myini(DMAREADABLEADDR);
	if (is_readable) printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, answer);
	put_user(answer, (int *) buffer);

    // Clean the result and set readable as false
    myouti(0, DMAANSADDR);
    myouti(0, DMAREADABLEADDR);

	return 0;
}


static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */

	// Write data into DMA
	char *data = kmalloc(sizeof(char) * ss, GFP_KERNEL);

	copy_from_user(data, buffer, ss);    // Copy data from user space
	struct DataIn *dataIn = (struct DataIn*) data;

	myoutc(dataIn->a, DMAOPCODEADDR);
    myouti(dataIn->b, DMAOPERANDBADDR);
    myouts(dataIn->c, DMAOPERANDCADDR);

	// Initialize queue work
	printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);
	INIT_WORK(work_routine, drv_arithmetic_routine);

	unsigned int IOMode = myini(DMABLOCKADDR);
	// Decide io mode
	if (IOMode) {      // Blocking IO: 1
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		flush_scheduled_work();

    } else {       // Non-locking IO : 0
	//	printk("%s,%s(): non-blocking\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
   	}

	kfree(data);
	return 0;
}


static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	/* Implement ioctl setting for your device */

	// Distinguish different cmd
	int ret;
	unsigned int is_readable;

	get_user(ret, (int *) arg);
	switch (cmd) {
		case HW5_IOCSETSTUID:      //Student ID
			myouti(ret, DMASTUIDADDR);
			if (ret >= 0) printk("%s:%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, ret);
			else return -1;
			break;
		case HW5_IOCSETRWOK:        //RW
			myouti(ret, DMARWOKADDR);
			if (ret == 1) printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
			else return -1;
			break;
		case HW5_IOCSETIOCOK:       //IOCTL
			myouti(ret, DMAIOCOKADDR);
			if (ret == 1) printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
			else return -1;
			break;
		case HW5_IOCSETIRQOK:      //IRQ
			myouti(ret, DMAIRQOKADDR);
			if (ret == 1) printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
			else return -1;
			break;
		case HW5_IOCSETBLOCK:       //Blocking or non-blocking
			myouti(ret, DMABLOCKADDR);
            if (ret == 1) printk("%s:%s(): Blocking IO\n",PREFIX_TITLE, __func__);
            else printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE,__func__);
			break;
		case HW5_IOCWAITREADABLE:       //Readable 
            is_readable = myini(DMAREADABLEADDR);
			printk("%s:%s(): wait readable 1\n", PREFIX_TITLE, __func__);
            while (!is_readable){
                msleep(1000);
                is_readable = myini(DMAREADABLEADDR);
            }
			put_user(is_readable, (int *) arg);
			break;
		default:
            return -1;
	}

	return 0;
}


int prime(int base, short nth) {
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }
    return num;
}


static irqreturn_t drv_interrupt_count(int irq, void *dev_id){
  unsigned int num;
  if (irq == IRQ_NUM) {
    num = myini(DMACOUNTADDR);
  	num += 1;
  	myouti(num, DMACOUNTADDR);
  }

  return IRQ_HANDLED;
}


static void drv_arithmetic_routine(struct work_struct* ws) {
	/* Implement arthemetic routine */
	struct DataIn dataIn;

    dataIn.a = myinc(DMAOPCODEADDR);
    dataIn.b = myini(DMAOPERANDBADDR);
    dataIn.c = myins(DMAOPERANDCADDR);

	int ans;
	if (dataIn.a == '+') ans = dataIn.b + dataIn.c;
	else if (dataIn.a == '-') ans = dataIn.b - dataIn.c;
	else if (dataIn.a == '*') ans = dataIn.b * dataIn.c;
	else if (dataIn.a == '/') ans = dataIn.b / dataIn.c;
	else if (dataIn.a == 'p') ans = prime(dataIn.b, dataIn.c);
	else ans = 0;

    printk("%s:%s(): %d %c %hd = %d \n", PREFIX_TITLE, __func__, dataIn.b, dataIn.a, dataIn.c, ans);

    myouti(ans, DMAANSADDR);   //Write the computation answer back
	myouti(1, DMAREADABLEADDR);    //When compution is done, change readable setting
}


static int __init init_modules(void) {
    
	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	/* Register chrdev */ 
	dev_t dev;
	int ret = 0;

	ret = alloc_chrdev_region(&dev, 0, 1, "mydev");
	if (ret) {
		printk("Cannot alloc chrdev\n");
		return ret;
	}
	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);
	printk("%s:%s(): register chrdev(%d,%d)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);

	/* Init cdev and make it alive */
	dev_cdevp = cdev_alloc();

	cdev_init(dev_cdevp, &fops);
	dev_cdevp->owner = THIS_MODULE;
	ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);
	if(ret < 0) {
		printk("Add chrdev failed\n");
		return ret;
	}

	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __func__);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	/* Allocate IRQ  */
    request_irq(IRQ_NUM, drv_interrupt_count, IRQF_SHARED, "IRQ_Count",  (void *)&drv_interrupt_count);

	return 0;
}

static void __exit exit_modules(void) {

    /* Print the number of interrupts */
	printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, myini(DMACOUNTADDR));

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	dev_t dev;
	
	dev = MKDEV(dev_major, dev_minor);
	cdev_del(dev_cdevp);

	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);
	unregister_chrdev_region(dev, 1);

	/* Free work routine */
	kfree(work_routine);

	/* Free Irq */
	free_irq(IRQ_NUM, (void *)&drv_interrupt_count);


	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);

}

module_init(init_modules);
module_exit(exit_modules);
