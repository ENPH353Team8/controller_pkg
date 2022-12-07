import os
# Function to rename multiple files
def main():
    i = 0
    path="/home/fizzer/cnn_trainer/data/"
    for filename in os.listdir(path):
        split = filename.split('_')
        my_source =path + filename
        num = int(split[0])+5800
        action = split[1][:-4]
        if action == 'F':
            action = '1'
        elif action == 'R':
            action = '0'
        else: 
            action = '2'
        my_dest =str(num) + '_' + action + ".jpg"
        my_dest =path + my_dest
        # rename() function will
        # rename all the files
        os.rename(my_source, my_dest)
        # print(my_dest)
# Driver Code
if __name__ == '__main__':
	# Calling main() function
	main()