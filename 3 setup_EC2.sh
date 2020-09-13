

# Run “source setup_EC2.sh” on command line to run this script
# Install python3
sudo yum install python3 -y

# Install anaconda
sudo yum install wget -y
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
chmod +x Anaconda3-2019.07-Linux-x86_64.sh

./Anaconda3-2019.07-Linux-x86_64.sh

# Conda env creation and activation
source ~/.bashrc
conda create --name py373 python=3.7.3
conda activate py373

# Install jupyter
pip install jupyter

# Start jupyter
jupyter notebook --no-browser --ip=0.0.0.0

# Get link for jupyter and change localhost with public IP of EC2 instance

