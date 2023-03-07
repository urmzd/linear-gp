#!/bin/bash

if [ "$1" = "unit" ]; then
    cargo test --no-fail-fast
elif [ "$1" = "iris" ]; then
    xvfb-run cargo run --example=iris
elif [ "$1" = "mountain_car" ]; then
    xvfb-run cargo run --example=mountain_car
elif [ "$1" = "cart_pole" ]; then
    xvfb-run cargo run --example=cart_pole
else
    echo "Unknown test suite: $1"
    exit 1
fi

if [ "$?" -ne 0 ]; then
    echo "Tests failed."
    exit 1
fi
