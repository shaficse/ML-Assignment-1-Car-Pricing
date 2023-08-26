FROM python:3.11.4-bookworm

WORKDIR /root/code

RUN pip3 install dash
RUN pip3 install dash_bootstrap_components
RUN pip install scikit-learn==1.2.2

CMD tail -f /dev/null