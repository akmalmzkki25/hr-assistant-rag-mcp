-- Data dummy untuk demo, bukan data karyawan sungguhan.
INSERT INTO employees (employee_id, name, position, manager_id) VALUES
    ('E001', 'Rina Wijaya', 'Engineering Manager', NULL),
    ('E002', 'Budi Santoso', 'Backend Engineer', 'E001'),
    ('E003', 'Sari Lestari', 'Data Analyst', 'E001')
ON CONFLICT (employee_id) DO NOTHING;

INSERT INTO leave_balances (employee_id, year, annual_quota, annual_used, sick_days_used) VALUES
    ('E001', 2026, 12, 5, 1),
    ('E002', 2026, 12, 10, 0),
    ('E003', 2026, 12, 0, 2)
ON CONFLICT (employee_id, year) DO NOTHING;
