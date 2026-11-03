# Local competition files

Download the competition files and put these three CSVs in this directory:

- `train.csv`: 1,200 rows; includes `SalePrice`.
- `test.csv`: 260 rows; no `SalePrice` column.
- `sample_soln.csv`: submission IDs 0–259. Only its `Id` column is used.

The test data's original IDs are 1201–1460. Submission rows follow the original test order and use the sample template's IDs.

`solns.csv` is an optional reference-answer file for the final audit section of the notebook. It is never an input to model fitting or blend-weight selection. Do not treat its zero-error copied submission as a trained-model result.

These CSVs are deliberately excluded from Git. Obtain them through the competition rather than redistributing them in this repository.
