FROM opencvcourses/opencv-docker:4.5.1

WORKDIR /app

RUN pip install Flask
RUN pip install requests
RUN pip install numpy
RUN pip install gunicorn

COPY . /app

ENTRYPOINT [ "python" ]

CMD ["__init__.py" ]