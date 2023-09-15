from augur.generate import generate

likelihood, sacc_data, tools = generate('/nethome/chisa002/augur/examples/config_test.yml',
                                        return_all_outputs=True, force_read=True)

sacc_data.save_fits("srd.fits", overwrite=True)

exit()
