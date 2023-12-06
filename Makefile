SHELL          = /bin/bash
PROJECT_NAME   = jax_xc
PROJECT_FOLDER = gen_repo tests third_party
PYTHON_FILES   = $(shell find . -type f -name "*.py" -not -path "./gen_repo/impl/prebuilt/**/*")
CPP_FILES      = $(shell find gen_repo -type f -name "*.h" -o -name "*.cc")
BAZEL_FILES    = $(shell find . -type f -name "*BUILD" -o -name "*.bzl")
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "Garena Online Private Limited"
BAZELOPT       =
PATH           := $(HOME)/go/bin:$(PATH)
ADDLICENSE_IGNORE = -ignore "gen_repo/impl/prebuilt/**/*" -ignore gen_repo/python.bzl

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, yapf)

cpplint-install:
	$(call check_install, cpplint)

clang-format-install:
	command -v clang-format-11 || sudo apt-get install -y clang-format-11

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.16 && sudo ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go)

bazel-install: go-install
	command -v bazel || (go install github.com/bazelbuild/bazelisk@latest && ln -sf $(HOME)/go/bin/bazelisk $(HOME)/go/bin/bazel)

buildifier-install: go-install
	command -v buildifier || go install github.com/bazelbuild/buildtools/buildifier@latest

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

auditwheel-install:
	$(call check_install_extra, auditwheel, auditwheel typed-ast)

# python linter

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --show-source --statistics

py-format: py-format-install
	yapf -r -d $(PYTHON_FILES)

# c++ linter

cpplint: cpplint-install
	cpplint $(CPP_FILES)

clang-format: clang-format-install
	clang-format-11 --style=file -i $(CPP_FILES) -n --Werror

# bazel file linter

buildifier: buildifier-install
	buildifier -r -lint=warn $(BAZEL_FILES)

# bazel build/test

bazel-debug: bazel-install
	bazel build $(BAZELOPT) //... --config=debug
	bazel run $(BAZELOPT) //:setup --config=debug -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-build: bazel-install
	bazel build $(BAZELOPT) //... --config=test
	bazel run $(BAZELOPT) //:setup --config=test -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-release: bazel-install
	bazel build $(BAZELOPT) //... --config=release
	bazel run $(BAZELOPT) //:setup --config=release -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-test: bazel-install
	bazel test --test_output=all $(BAZELOPT) //... --config=test --spawn_strategy=local --color=yes

bazel-clean: bazel-install
	bazel clean --expunge

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2023 $(ADDLICENSE_IGNORE) -check $(PROJECT_FOLDER)

docstyle: doc-install
	pydocstyle $(PROJECT_NAME) && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

doc-clean:
	cd docs && make clean

doc-benchmark:
	pandoc benchmark/README.md --from markdown --to rst -s -o docs/content/benchmark.rst --columns 1000
	cd benchmark && ./plot.py --suffix png && mv *.png ../docs/_static/images/throughput

lint: buildifier flake8 py-format clang-format cpplint # docstyle

format: py-format-install clang-format-install buildifier-install addlicense-install
	yapf -ir $(PYTHON_FILES)
	clang-format-11 -style=file -i $(CPP_FILES)
	buildifier -r -lint=fix $(BAZEL_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2022 $(ADDLICENSE_IGNORE) $(PROJECT_FOLDER)

# Build docker images

docker-dev:
	docker build --network=host -t $(PROJECT_NAME):$(COMMIT_HASH) -f docker/dev.dockerfile .
	docker run --network=host -v /:/host -it $(PROJECT_NAME):$(COMMIT_HASH) bash
	echo successfully build docker image with tag $(PROJECT_NAME):$(COMMIT_HASH)

# for mainland China
docker-dev-cn:
	docker build --network=host -t $(PROJECT_NAME):$(COMMIT_HASH) -f docker/dev-cn.dockerfile .
	docker run --network=host -v /:/host -it $(PROJECT_NAME):$(COMMIT_HASH) bash
	echo successfully build docker image with tag $(PROJECT_NAME):$(COMMIT_HASH)

docker-release:
	docker build --network=host -t $(PROJECT_NAME)-release:$(COMMIT_HASH) -f docker/release.dockerfile .
	mkdir -p wheelhouse
	docker run --network=host -v `pwd`/wheelhouse:/whl -it $(PROJECT_NAME)-release:$(COMMIT_HASH) bash -c "cp wheelhouse/* /whl"
	echo successfully build docker image with tag $(PROJECT_NAME)-release:$(COMMIT_HASH)

pypi-wheel: auditwheel-install bazel-release
	ls dist/*.whl -Art | tail -n 1 | xargs auditwheel repair --plat manylinux_2_17_x86_64

release-test1:
	cd envpool && python3 make_test.py

release-test2:
	cd examples && python3 make_env.py && python3 env_step.py

release-test: release-test1 release-test2
