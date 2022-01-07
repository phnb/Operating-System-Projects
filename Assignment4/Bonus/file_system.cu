#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 dir = 1025;   //pointer points to parent dictionary: 1025 represents the root directory
__device__ __managed__ bool is_directory = 0;     //Indicate whether the created file is directory

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
      size: 2 bytes;
      dictionary: 2 bytes;
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
      if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] == dir) {
        //printf("already exists FCB: %d \n", i);
        return i;
      }
    }
  }

  //If the file doesn't exists
  //Check whether it's a directory
  bool is_dir = is_directory;
  is_directory = 0;
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
  *(short *)&fs->volume[base] = 0;    //Initialize the size of file as 0 bytes
  *(short *)&fs->volume[base + 2] = dir;    //Initialize the pointer pointing to parent directory
  *(int *)&fs->volume[base + 4] = is_dir ? 32769 : target_block;    //Initialize the starting address of allocated block
  *(short *)&fs->volume[base + 8] = gtime;     //Initialize the modified time of file
  *(short *)&fs->volume[base + 10] = gtime;     //Initialize the created time of file

  //Update the size of corresponding parent directory
  int dir_fp = dir;
  index++;
  if (dir_fp != 1025) {
    *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + dir_fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] += index;
  }
  return num;
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
  //Update the parent directory modified time
  if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] != 1025) {
    short index = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2];
    *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8] = gtime;
    if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] != 1025) {
      index = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2];
      *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + index * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8] = gtime;
    }
  }

  //If the current block is empty
  int new_block = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
  if (fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] == 0) {
    int start_block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
    int start_address = fs-> FILE_BASE_ADDRESS + start_block * fs->STORAGE_BLOCK_SIZE;
    for (int i = 0; i < size; i++) {
      fs->volume[start_address + i] = input[i];
    }
    //Update information: size
    *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] = size;
    //Updata bitmap
    for (int i = 0; i < new_block; i++) {
      fs->volume[(start_block + i) / 8] = fs->volume[start_block / 8] | (1 << ((start_block + i) % 8));
    }


  //Otherwies the current block isn't empty
  } else {
    short old_size = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
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
      *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] = size;
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
      *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] = size;
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
  int a[1024];
  for (int i = 0; i < fs->MAX_FILE_NUM; i++) {
    if (fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE] != 0) {  //Check whether it's empty
      if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] == dir) {  //Check whether it's under current directory
        a[num] = i;
        num++;
      }
    }
  }

  //Use insertion sort to sort the file
  //op == LS_D
  if (op == 0) {
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
      bool is_dir = false;
      int k = 0;
      for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
        u32 base = fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + j;
        if (fs->volume[base] != '\0') {
          file_name[j] = fs->volume[base];
        } else {
          file_name[j] = '\0';
          break;
        }
      }
      if (*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] == 32769) is_dir = true;
      if (is_dir) printf("%s d\n", file_name);
      else printf("%s\n", file_name);
    }
  //op == LS_S
  } else if (op == 1) {
    printf("===sort by file size===\n");
    //Sort the files by file size
    for (int i = 1; i < num; i++) {
      short size = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
      int temp1 = a[i];
      int j = i - 1;
      while (j >= 0 && (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + a[j] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] < size)) {
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
      bool is_dir = false;
      int k = 0;
      for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
        u32 base = fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + j;
        if (fs->volume[base] != '\0') {
          file_name[j] = fs->volume[base];
        } else {
          file_name[j] = '\0';
          break;
        }
      }
      short size = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
      if (*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + a[i] * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] == 32769) is_dir = true;
      if (is_dir) printf("%s %d d\n", file_name, size);
      else printf("%s %d\n", file_name, size);
    }
  //op == CD_P
  } else if (op == 5) {
    if (dir == 1025) {
      printf("here is the root directory!\n");
    } else {
      dir = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + dir * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2];
    }
  //op == PWD
  } else if (op == 7) {
    if (dir == 1025){
      printf("/\n");
    } else {
      int dir_fp = dir;
      //Regroup the file name
      char file_name1[20];
      for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
        u32 base = fs->SUPERBLOCK_SIZE + dir_fp * fs->FCB_SIZE + i;
        if (fs->volume[base] != '\0') {
          file_name1[i] = fs->volume[base];
        } else {
          file_name1[i] = '\0';
          break;
        }
      }
      dir_fp = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + dir_fp * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2];
      if (dir_fp == 1025) {
        printf("/%s\n", file_name1);
      } else {
        char file_name2[20];
        for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
          u32 base = fs->SUPERBLOCK_SIZE + dir_fp * fs->FCB_SIZE + i;
          if (fs->volume[base] != '\0') {
            file_name2[i] = fs->volume[base];
          } else {
            file_name2[i] = '\0';
            break;
          }
        }
        printf("/%s/%s \n", file_name2, file_name1);
      }
    }
  }
}


__device__ void rm(FileSystem *fs, char *s) 
{
/* Implement rm operation here */
  //Determine the location of the delete file/directory
  u32 delete_file;
  for (int i = 0; i < fs->MAX_FILE_NUM; i++) {
    u32 base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
    u32 index = 0;
    while (s[index] != '\0') {
      if (s[index] != fs->volume[base + index]) break;
      index++;
    }
    if ((s[index] == '\0') && (fs->volume[base + index] == '\0')) {
      if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] == dir) {
        delete_file = i;
        break;
      }
    }
  }

  // Check whether it's a file
  bool is_dir = false;
  if (*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + delete_file * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] == 32769) is_dir = true;
  if (is_dir) {
    printf("You cannot use RM to delete a directory!");
    return;
  }

  //Update corresponding information
  short size = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + delete_file * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
  int start_block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + delete_file * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
  int start_address = fs-> FILE_BASE_ADDRESS + start_block * fs->STORAGE_BLOCK_SIZE;
  short length = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
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

  //Calculate the number of rest files and update the starting address of blocks (if need)
  u32 rest_files = 0;
  for (int i = delete_file; i < fs->MAX_FILE_NUM - 1; i++) {
    if ( fs->volume[fs->SUPERBLOCK_SIZE + (i + 1) * fs->FCB_SIZE] != 0 ) {
      rest_files++;
    }
  }
  for (int i = 0; i < fs->MAX_FILE_NUM; i++) {
    int block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
    if (block == 32769) continue; 
    if (*(int *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] > start_block) {
      *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] -= length;
    }
  }
  //Move the bits in the bitmap
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
    fs->volume[i] = 0;
  }
  for (int i = 0; i < (delete_file + 1 + rest_files); i++) {
    int block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
    if (block == 32769) continue;  //It's a directory
    short size = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
    short num = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
    for (int j = 0; j < num; j++) {
      fs->volume[(block + j) / 8] = fs->volume[block / 8] | (1 << (block + j) % 8);
    }
  }
  //Move the file content forward
  int address = start_address;
  for (int i = 0; i < rest_files; i++) {
    int block = *(int *)&fs->volume[fs->SUPERBLOCK_SIZE + (i + delete_file) * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
    if (block == 32769) continue;  //It's a directory
    short size = *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + (i + delete_file + 1) * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
    short num = (size % 32 == 0) ? size / 32 : (size / 32 + 1);
    for (int j =  0; j < num * fs->STORAGE_BLOCK_SIZE; j++) {
      fs->volume[j + address] = fs->volume[j + address + length * fs->STORAGE_BLOCK_SIZE];
    }
    address += num * fs->STORAGE_BLOCK_SIZE;
  }
}


__device__ void mkdir(FileSystem *fs, char *s) 
{ 
  is_directory = 1;
  u32 fp = fs_open(fs, s, G_WRITE);
}

__device__ void cd(FileSystem *fs, char *s) 
{
  int i = 0;
  bool is_dir = false;
  // Check whether it's in the file system
  for (; i < fs->MAX_FILE_NUM; i++) {
    u32 base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
    u32 index = 0;
    while (s[index] != '\0') {
      if (s[index] != fs->volume[base + index]) break;
      index++;
    }
    if ((s[index] == '\0') && (fs->volume[base + index] == '\0')){
      if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] == dir) break;
    }
  }
  if (*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] == 32769) is_dir = true;
  if (is_dir) dir = i;
  else printf("Incorrect directory name!");
}

__device__ void rm_rf(FileSystem *fs, char *s) 
{
  int i = 0;
  bool is_dir = false;
  bool is_par = false;
  int dir_fp = dir;
  int length;
  // Check whether it's in the file system
  for (; i < fs->MAX_FILE_NUM; i++) {
    u32 base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
    u32 index = 0;
    while (s[index] != '\0') {
      if (s[index] != fs->volume[base + index]) break;
      index++;
      length++;
    }
    if (s[index] == '\0') {
      if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] == dir) {
        length = index + 1;
        break;
      }
    }
  }
  if (*(int*)&fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] == 32769) is_dir = true;
  if (!is_dir) {
    printf("You can only use rm_rf to delete a directory!");
    return;
  }

  //Delete all files under this directory
  for (int k = 0; k < fs->MAX_FILE_NUM; k++) {
    if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + k * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] == i) {
      for (int m = 0; m < fs->MAX_FILE_NUM; m++) {
        if (*(short *)&fs->volume[fs->SUPERBLOCK_SIZE + m * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 2] == k) {
          //Regroup the file name
          char file_name[20];
          for (int t = 0; t < fs->MAX_FILENAME_SIZE; t++) {
            u32 base = fs->SUPERBLOCK_SIZE + m * fs->FCB_SIZE + t;
            if (fs->volume[base] != '\0') {
              file_name[t] = fs->volume[base];
            } else {
              file_name[t] = '\0';
              break;
            }
          }
          dir = k;
          rm(fs, file_name); //Delete the file
          is_par = true;
        } 
      }
      //If it's a directory (clear the FCB)
      if (is_par) { 
        for (int t = 0; t < fs->FCB_SIZE; t++) {
          fs->volume[fs->SUPERBLOCK_SIZE + k * fs->FCB_SIZE + t] = 0;
        }
      } else {
        //If it's a file, call the rm() function
        char file_name[20];
        for (int t = 0; t < fs->MAX_FILENAME_SIZE; t++) {
          u32 base = fs->SUPERBLOCK_SIZE + k * fs->FCB_SIZE + t;
          if (fs->volume[base] != '\0') {
            file_name[t] = fs->volume[base];
          } else {
            file_name[t] = '\0';
            break;
          }
        }
        dir = i;
        rm(fs, file_name); //Delete the file
      }
    }
  }
  //Delete the directory (clear the FCB)
  for (int t = 0; t < fs->FCB_SIZE; t++) {
    fs->volume[fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE + t] = 0;
  }
  dir = dir_fp;
  //Update the size of corresponding parent directory
  if (dir != 1025) {
    *(short *)&fs->volume[fs->SUPERBLOCK_SIZE + dir * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] -= length;
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  if (op == 2) rm(fs, s);
  else if (op == 3) mkdir(fs, s);
  else if (op == 4) cd(fs, s);
  else if (op == 6) rm_rf(fs, s);
  else printf("Invalid Command!\n");
}
