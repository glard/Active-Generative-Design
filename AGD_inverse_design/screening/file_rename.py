import os

# Function to rename multiple files
def main():

    for count, filename in enumerate(os.listdir("CIF")):
        # txt = "2-1-mp-32554.cif"
        try:
            x = filename.split("-")

            print(x)

            dst = x[2] + "-" + x[3]

            print(dst)

            src ='./CIF/'+ filename
            dst ='./renamed_cif/'+ dst

            # rename() function will
            # rename all the files
            os.rename(src, dst)
        except:
            continue

# Driver Code
if __name__ == '__main__':

    # Calling main() function
    main()
