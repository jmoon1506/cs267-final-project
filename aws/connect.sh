#!/bin/bash
ssh -Y -i parallel.pem \
    -L 8888:localhost:8888 \
	-L 8889:localhost:8889 \
	-L 8890:localhost:8890 \
	-L 8891:localhost:8891 \
	-L 8892:localhost:8892 \
ubuntu@$1