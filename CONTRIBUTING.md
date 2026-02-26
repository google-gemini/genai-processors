# How to Contribute

We welcome contributions of new Processors that can be shared with the
community. Please add your new Processors under the
[`contrib/`](genai_processors/contrib) folder.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project.

Visit <https://cla.developers.google.com/> to see your current agreements on
file or to sign a new one.

You generally only need to submit a CLA once. If you've already submitted one
(even for a different Google project), you likely don't need to do it again.

## Code reviews

All submissions, including those from project members, require code review. We
use GitHub pull requests for this process.

For more information on using pull requests, please consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/).

If you would prefer to have more flexibility and avoid bandwidth of GenAI
Processors maintainers becoming a bottleneck, consider keeping your processor in
a separate repository. We would be happy to add a link to your repository from
[contrib/README.md](genai_processors/contrib/README.md) to increase visibility.

## File structure

Please use the following file structure:

 * genai_processors/contrib/
   * **your_processor.py** - Implementation of the processor.
   * **your_processor.md** - Documentation.
   * tests/
     * **your_processor_test.py** - Tests.

If your processor is more complex and would benefit from being split into
several files, you can create a `genai_processors/contrib/your_processor/`
sub-directory. Also consider hosting it in a separate repository and adding a
link instead.

## Community Guidelines

This project adheres to
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
