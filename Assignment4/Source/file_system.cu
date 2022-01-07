#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  for (int i = 0; i < fs->FILE_BASE_ADDRESS; i++) {
    fs->volume[fs->SUPERBLOCK_SIZE + i] = 0;
  }
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here: 
    Design of FCB 32 bytes (in order):
      name: 20 bytes;
      size: 4 bytes;
      starting address: 4 bytes;    #number of blocks
      modified time: 2 bytes;
      created time: 2 bytes;
  */

  //Check whether the file exists by name
  for (int i = 0; i < fs->MAX_FILE_NUM; i++) {
    u32 base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
    u32 index = 0;
    while (s[index] != '\0') {
      if (s[index] != fs->volume[base + index]) break;
      index++;
    }
    if ((s[index] == '\0') && (fs->volume[base + index] == '\0')) {
      return i;
    }
  }

  if (op == 1) {   //Write mode
    //If the file doesn't exists
    //Find the none-occuiped place for new created file (use bitmap)
    int target_block = -1;
    for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
      for (int j = 0; j < 8; j++) {
        if ((fs->volume[i] >> j) == 0) {
          target_block = i * 8 + j;
          break;
        }
      }
      if (target_block != -1) break;
    }
    //Create a new FCB and initialize it
    u32 index = 0;
    u32 num = 0;
    for (; num < fs->FCB_ENTRIES; num++) {
      if (fs->volume[fs->SUPERBLOCK_SIZE + num * fs->FCB_SIZE] == 0) break;
    }
    //Initialize the name of file
    while (s[index] != '\0') {
      fs->volume[fs->SUPERBLOCK_SIZE + num * fs->FCB_SIZE + index] =  s[index]; 
      if (index >= fs->MAX_FILENAME_SIZE - 1) break;
      index++;
    }
    fs->volume[fs->SUPERBLOCK_SIZE + num * fs->FCB_SIZE + index] =  '\0';
    int base = fs->SUPERBLOCK_SIZE + num * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE;
    *(int *)&fs->volume[base] = 0;    //Initialize the size of file as 0 bytes
    *(int *)&fs->volume[base + 4] = target_block;    //Initialize the starting address of allocated block
    *(short *)&fs->volume[base + 8] = gtime;     //Initialize the modified time of file
    *(short *)&fs->volume[base + 10] = gtime;     //Initialize the created time of file
    return num;
  } else {
    printf("no such file!\n");
  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  for (u32 i = 0; i < size; i++) {
    int start_address = fs->FILE_BASE_ADDRESS + *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] * fs->STORAGE_BLOCK_SIZE;
    output[i] = fs->volume[start_address + i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  //If the current block is empty
  int new_block = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
  if (fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] == 0) {
    int start_block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
    int start_address = fs-> FILE_BASE_ADDRESS + start_block * fs->STORAGE_BLOCK_SIZE;
    for (int i = 0; i < size; i++) {
      fs->volume[start_address + i] = input[i];
    }
    //Update information: size
    *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] = size;
    //Updata bitmap
    for (int i = 0; i < new_block; i++) {
      fs->volume[(start_block + i) / 8] = fs->volume[start_block / 8] | (1 << ((start_block + i) % 8));
    }
  //Otherwies the current block isn't empty
  } else {
    int old_size = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
    int old_block = (old_size % 32 == 0) ? old_size / 32 : (old_size / 32 + 1);
    //If the number of the blocks remains the same
    if (old_block == new_block) {
      int start_address = fs-> FILE_BASE_ADDRESS + *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] * fs->STORAGE_BLOCK_SIZE;
      for (int i = 0; i < old_size; i++) {
        fs->volume[start_address + i] = NULL;
      }
      for (int i = 0; i < size; i++) {
        fs->volume[start_address + i] = input[i];
      }
      //Update information: size and modified time
      *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] = size;
      *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8] = gtime;

    //If the number of the blocks increases or decreases
    } else {
      //Regroup the file name
      char file_name[20];
      for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
        u32 base = fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + i;
        if (fs->volume[base] != '\0') {
          file_name[i] = fs->volume[base];
        } else {
          file_name[i] = '\0';
          break;
        }
      }
      short create_time = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 10];

      //Clean the old FCB and content, find a new place
      fs_gsys(fs, RM, file_name);
      u32 new_fp = fs_open(fs, file_name, G_WRITE);

      //Write the FCB clock (update the file size, modifies time and create time)
      *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] = size;
      *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8] = gtime;
      *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 10] = create_time;
      //Write data into the file
      int start_block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
      int start_address = fs-> FILE_BASE_ADDRESS + start_block * fs->STORAGE_BLOCK_SIZE;
      for (int i = 0; i < size; i++) {
        fs->volume[start_address + i] = input[i];
      }
      //Write the bitmap (update)
      for (int i = 0; i < new_block; i++) {
        fs->volume[(start_block + i) / 8] = fs->volume[start_block / 8] | (1 << ((start_block + i) % 8));
      }
    }
  }
  gtime++;
  return size;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  int num = 0;
  for (int i = 0; i < fs->MAX_FILE_NUM; i++) {
    if (fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE] != 0) num++;
  }
  int a[1024];
  for (int i = 0; i < num; i++) {
    a[i] = i;
  }

  //Use insertion sort to sort the file
  if (op == LS_D) {
    //Sort the files by modified time
    printf("===sort by modified time===\n");
    for (int i = 1; i < num; i++) {
      int temp = a[i];
      short time = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8];
      int j = i - 1;
      while (j >= 0 && (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + a[j] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8] < time)) {
        a[j+1] = a[j];
        j = j - 1;
      }
      a[j+1] = temp;
    }	


    //Print the files
    for (int i = 0; i < num; i++) {
      char file_name[20];
      for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
        u32 base = fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + j;
        if (fs->volume[base] != '\0') {
          file_name[j] = fs->volume[base];
        } else {
          file_name[j] = '\0';
          break;
        }
      }
      printf("%s\n", file_name);
    }
  } else if (op == LS_S) {
    printf("===sort by file size===\n");
    //Sort the files by file size
    for (int i = 1; i < num; i++) {
      int size = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
      int temp1 = a[i];
      int j = i - 1;
      while (j >= 0 && (*(int *)&fs->volume[fs->SUPERBLOCK_SIZE + a[j] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] < size)) {
        a[j+1] = a[j];
        j = j - 1;
        
      }
      a[j+1] = temp1;
    }
    //Print the files
    for (int i = 0; i < num; i++) {
      if (i < num -1) {   //If they have the same size (judge by modifed time)
        int size1 = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
        int size2 = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + a[i+1] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
        if (size1 == size2) {
          int len = 0;
          int p = i;
          while (*(int *)&fs->volume[fs->SUPERBLOCK_SIZE + a[p] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] == size1) {
            len++;
            p++;
            if (p == num) break;
          }
          for (int q = i+1; q < i+len; q++) {
            int temp2 = a[q];
            short time = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + a[q] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 10];
            int qq = q - 1;
            while (qq >= i && (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + a[qq] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 10] > time)) {
              a[qq+1] = a[qq];
              qq = qq - 1;
            }
            a[qq+1] = temp2;
          }
        }
      }   
    
      char file_name[20];
      for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
        u32 base = fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + j;
        if (fs->volume[base] != '\0') {
          file_name[j] = fs->volume[base];
        } else {
          file_name[j] = '\0';
          break;
        }
      }
      int size = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
      printf("%s %d\n", file_name, size);
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  //Determine the location of the delete file
  u32 delete_file;
  for (int i = 0; i < fs->MAX_FILE_NUM; i++) {
    u32 base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
    u32 index = 0;
    while (s[index] != '\0') {
      if (s[index] != fs->volume[base + index]) break;
      index++;
    }
    if ((s[index] == '\0') && (fs->volume[base + index] == '\0')) {
      delete_file = i;
      break;
    }
  }

  //Update corresponding information
  int size = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + delete_file * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
  int start_block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + delete_file * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
  int start_address = fs-> FILE_BASE_ADDRESS + start_block * fs->STORAGE_BLOCK_SIZE;
  int length = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
  //Update the bitmap
  for (int i = 0; i < length; i++) {
    fs->volume[(start_block + i) / 8] = fs->volume[start_block / 8] & (~(1 << ((start_block + i) % 8)));
  }
  //Clear file content
  for (int i = 0; i < size; i++) {
    fs->volume[start_address + i] = NULL;
  }
  //Clear FCB content
  for (int i = 0; i < fs->FCB_SIZE; i++) {
    fs->volume[fs->SUPERBLOCK_SIZE + delete_file * fs->FCB_SIZE + i] = 0;
  }

  //Move the FCB blocks
  u32 rest_files = 0;
  for (int i = delete_file; i < fs->MAX_FILE_NUM - 1; i++) {
    if ( fs->volume[fs->SUPERBLOCK_SIZE + (i + 1) * fs->FCB_SIZE] != 0 ) {
      for (int j = 0; j < fs->FCB_SIZE; j++) {
        fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + j] = fs->volume[fs->SUPERBLOCK_SIZE + (i + 1) * fs->FCB_SIZE + j];
      }
      *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] -= length;
      rest_files++;
    }
  }
  for (int i = 0; i < fs->FCB_SIZE; i++) {
    fs->volume[fs->SUPERBLOCK_SIZE + (delete_file + rest_files) * fs->FCB_SIZE + i] = 0;
  }
  //Move the bits in the bitmap
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
    fs->volume[i] = 0;
  }
  for (int i = 0; i < (delete_file + rest_files); i++) {
    int block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
    int size = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
    int num = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
    for (int j = 0; j < num; j++) {
      fs->volume[(block + j) / 8] = fs->volume[start_block / 8] | (1 << (block + j) % 8);
    }
  }
  //Move the file content forward
  int address = start_address;
  for (int i = 0; i < rest_files; i++) {
    int size = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + (i + delete_file + 1) * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
    int num = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
    for (int j =  0; j < num * fs->STORAGE_BLOCK_SIZE; j++) {
      fs->volume[j + address] = fs->volume[j + address + length * fs->STORAGE_BLOCK_SIZE];
    }
    address += num * fs->STORAGE_BLOCK_SIZE;
  }
}
