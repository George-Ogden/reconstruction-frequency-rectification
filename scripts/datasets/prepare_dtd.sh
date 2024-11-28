#!/usr/bin/env bash

set -x

find datasets/dtd/labels/train* -exec sh -c "cat {} >> datasets/dtd/train_list.txt" \;
find datasets/dtd/labels/val* -exec sh -c "cat {} >> datasets/dtd/train_list.txt" \;
find datasets/dtd/labels/test* -exec sh -c "cat {} >> datasets/dtd/test_list.txt" \;