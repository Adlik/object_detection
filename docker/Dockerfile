FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update -y && \
    apt-get install -y curl vim git openssh-client openssh-server libgl1-mesa-glx&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Allow openSSH password-free login between containers
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking | grep -v Port > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    echo "    Port 10001" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    cat /etc/ssh/sshd_config | grep -v Port > /etc/ssh/sshd_config.new && \
    echo "Port 10001" >> /etc/ssh/sshd_config.new && \
    mv /etc/ssh/sshd_config.new /etc/ssh/sshd_config && \
    ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa && \
    sed -i -E -e 's/^PermitRootLogin\s+prohibit-password$/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create working directory
RUN mkdir -p /root/work
WORKDIR /root/work

